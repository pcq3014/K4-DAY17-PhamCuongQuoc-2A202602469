# Báo cáo Day 17: Memory Systems for AI Agent

Sinh viên: Phạm Cường Quốc (2A202602469)

Cách chạy lại toàn bộ kết quả trong báo cáo (offline, không cần API key, kết quả lặp lại được):

```bash
pip install -r requirements.txt
python src/benchmark.py          # thêm -v để in từng câu trả lời recall
pytest src/test_agents.py -v
```

Muốn chạy với LLM thật thì đặt `LLM_PROVIDER`, `LLM_MODEL` và key tương ứng trong `.env`, sau đó chạy `python src/benchmark.py --live`.

## 1. Kiến trúc

| Lớp memory | Baseline | Advanced | Nằm ở đâu |
|---|---|---|---|
| Short-term (trong thread) | Toàn bộ message của thread, gửi lại mỗi lượt | Các message gần nhất (`compact_keep_messages=4`) | `SessionState` / `CompactMemoryManager` |
| Persistent (qua session) | Không có | `state/profiles/<user>/User.md` | `UserProfileStore` |
| Compact (nén lịch sử) | Không có | Tóm tắt có giới hạn (≤ 6 dòng) khi thread vượt 800 token | `CompactMemoryManager._compact()` |

Mỗi lượt của Advanced Agent:

```
message → extract_fact_candidates()  (gán confidence cho từng fact)
        → apply_candidates()         (lọc theo ngưỡng, xử lý mâu thuẫn, ghi User.md)
        → CompactMemoryManager.append()  (tự compact khi vượt ngưỡng)
        → prompt = system + profile view + summary + recent messages
        → trả lời → cập nhật bộ đếm token
```

Có hai điểm thiết kế cần nói rõ:

- **File lưu trữ khác với phần đưa vào prompt.** `User.md` giữ metadata để audit (confidence, số lần nhắc, revision, log đính chính). Prompt chỉ nhận `prompt_view()`, tức các fact hiện tại. Ở lần đầu, khi đưa nguyên file vào prompt, prompt tokens của Advanced trên Standard là **+166%** so với Baseline. Đổi sang profile view thì còn **+83%**.
- **Baseline công bằng.** Baseline dùng cùng extractor và cùng hàm trả lời, chỉ khác nguồn nhớ là message của chính thread đó. Vì vậy trong cùng thread nó vẫn nhớ (có test kiểm chứng). Sang thread mới thì nó mất hết.

Live mode (`--live`): Baseline dùng `create_agent` + `InMemorySaver`. Advanced dùng `create_agent` với:

- tool `read_user_memory` và `remember_user_fact`
- `dynamic_prompt` để chèn profile và summary
- `SummarizationMiddleware`

Fact do LLM đề xuất qua tool cũng phải qua cùng ngưỡng confidence. Provider hỗ trợ: openai, custom, gemini, anthropic, ollama, openrouter (`model_provider.py`).

## 2. Kết quả benchmark (offline, deterministic)

Cấu hình: `compact_threshold=800 tokens · keep=4 messages · confidence_threshold=0.6`

### Standard Benchmark (`conversations.json`: 10 hội thoại, 101 lượt, 14 câu recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|:--------------------:|:----------------:|----------------------:|------------:|
| Baseline |             2,623 |                  16,536 | 0%                   | 1.50/5           |                     0 |           0 |
| Advanced |             2,716 |                  30,263 | 100%                 | 5.00/5           |                 1,069 |           0 |

### Long-Context Stress Benchmark (`advanced_long_context.json`: 1 hội thoại, 16 lượt dài, 3 câu recall)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|:--------------------:|:----------------:|----------------------:|------------:|
| Baseline |             2,552 |                  22,237 | 0%                   | 1.50/5           |                     0 |           0 |
| Advanced |             2,914 |                  11,033 | 100%                 | 5.00/5           |                   582 |           4 |

Prompt load từng lượt trên thread stress (token đưa vào prompt ở mỗi lượt):

| Lượt | 1 | 4 | 5 | 8 | 11 | 14 | 16 |
|---|---|---|---|---|---|---|---|
| Baseline | 229 | 709 | 876 | 1,341 | 1,731 | 2,182 | 2,469 |
| Advanced | 307 | 807 | **493** ← compact #1 | 591 ← #2 | 549 ← #3 | 543 ← #4 | 822 |

Prompt của Baseline tăng tuyến tính theo từng lượt, nên tổng token qua cả thread tăng theo bậc hai (O(n²)). Advanced tăng lên tới khoảng 800 token, compact, rồi tụt về khoảng 500. Đường đi có dạng răng cưa và bị chặn trên.

## 3. Phân tích

**Vì sao Advanced có recall tốt hơn (0% → 100%).** Câu recall được hỏi ở thread mới. Baseline không có gì để đọc, nên nó trả lời "chưa có trong bộ nhớ". Đây là hành vi đúng: nó không giả vờ nhớ. Advanced đọc `User.md`, nơi fact đã được chuẩn hóa thành field. Các correction cũng được giữ đúng:

- Nơi ở Đà Nẵng → Huế (conv-03), rồi Huế → Đà Nẵng trong stress test.
- Nghề backend → MLOps engineer (conv-06).
- Nhiễu bị loại: "product manager" chỉ là câu đùa, "Hà Nội" chỉ là nơi đi họp, "nhắc lại Đà Nẵng như ví dụ cũ" không được tính là nơi ở.

**Vì sao Advanced tốn hơn ở hội thoại ngắn (prompt +83%, agent tokens +4%).** Mỗi lượt Advanced phải mang theo system prompt dài hơn cộng với profile (khoảng 60–120 token). Một hội thoại 10 lượt ngắn chỉ có khoảng 450 token lịch sử, chưa chạm ngưỡng compact. Nghĩa là Advanced trả chi phí cố định của memory mà chưa được hưởng lợi từ compact. Agent tokens cũng cao hơn vì việc ghi `User.md` và sinh summary là output thật. Kết luận: với chat ngắn, phần đáng tiền của Advanced là **recall qua session**, không phải tiết kiệm token.

**Vì sao compact giúp Advanced thắng ở hội thoại dài (prompt −50%).** Compact không làm giảm số token người dùng gửi vào, cũng không giảm câu trả lời. Agent tokens của Advanced thậm chí cao hơn 14% vì phải sinh summary. Thứ compact tối ưu là **prompt tokens processed**, tức phần ngữ cảnh cũ bị gửi lại ở mọi lượt. Lịch sử cũ được thay bằng summary tối đa 6 dòng cộng với 4 message gần nhất, nên prompt mỗi lượt bị chặn ở khoảng threshold, không còn tăng theo độ dài thread. Thread càng dài thì chênh lệch càng lớn.

**Summary có làm mất thông tin không.** Có. Summary heuristic chỉ giữ câu đầu của mỗi lượt cũ, nên chi tiết như "Mach 1.1" hay "29.500 feet" bị mất sau compact. Recall vẫn đạt 100% vì fact ổn định đã được tách sang `User.md` **trước** khi message bị nén. Đây chính là lý do cần tách bạch persistent và compact: compact được phép mất chi tiết hội thoại, còn persistent thì không được mất fact.

**Memory growth và rủi ro.** `User.md` của user `dungct` lớn lên 1,069 bytes sau 10 phiên, của user stress là 582 bytes. Có ba nhóm rủi ro:

1. *Phình to.* File tăng theo số fact và topic. Phần interests không có giới hạn trong file, nhưng chỉ top-4 (đã xếp hạng có decay) được đưa vào prompt. Log đính chính bị giới hạn 5 dòng.
2. *Lưu sai fact.* Đây là lỗi nguy hiểm nhất vì nó tồn tại qua mọi session. Hiện được giảm bằng ngưỡng confidence, phát hiện câu hỏi, và kiểm tra phủ định/nhiễu (xem mục 4).
3. *Fact cũ sai lọt vào prompt.* Giá trị cũ chỉ nằm trong mục Corrections (để audit), không bao giờ vào `prompt_view()`.

Ngoài ra extractor dựa trên rule, nên với cách diễn đạt nằm ngoài pattern (ví dụ "mình dọn về Sài Gòn rồi") thì recall sẽ giảm. Live mode bù lại phần này bằng tool `remember_user_fact`, nhưng đổi lại phải tin vào LLM hơn.

## 4. Bonus đã làm

| Bonus | Cách làm | Giải quyết vấn đề gì | Rủi ro thêm vào |
|---|---|---|---|
| **Confidence threshold** | Mỗi fact có confidence. Câu chứa nhiễu ("đùa", "đi họp", "công tác", "ví dụ cũ", "đừng nói") còn 0.2. Câu giả định ("nếu", "có lẽ", "cân nhắc") bị trừ 0.3. Câu đính chính ("đính chính", "giờ", "chuyển sang") được +0.05. Chỉ ghi khi ≥ `MEMORY_CONFIDENCE_THRESHOLD` (mặc định 0.6). Câu hỏi và yêu cầu nhắc lại không bao giờ được ghi. | Tránh ghi "product manager" (câu đùa), "Hà Nội" (đi công tác), "có lẽ chuyển ra Hà Nội" (0.55) vào profile. | Ngưỡng cao quá thì bỏ sót fact thật diễn đạt rụt rè. Cue list là tiếng Việt viết tay, có thể sai với văn phong khác. |
| **Conflict handling** | Với cùng một key, fact đủ tin cậy và mới nhất sẽ thay fact cũ. Giá trị cũ vào log `Corrections` (giới hạn 5 dòng). Trong một câu, mention bị phủ định ("không còn", "chứ không", "lúc đầu", "trước đó") bị bỏ qua và mention cuối cùng còn hợp lệ được chọn. Style được merge theo slot (độ dài / bullet / ví dụ / trade-off), không ghi đè cả cụm. | Trả lời "Huế"/"Đà Nẵng" đúng thời điểm, "MLOps engineer" thay vì "backend engineer". Test `test_cross_session_recall` kiểm tra fact cũ không lọt vào câu trả lời. | "Mới nhất thắng" có thể sai khi người dùng kể chuyện quá khứ bằng thì hiện tại. |
| **Entity extraction** | Fact có cấu trúc: `name, location, profession, drink, food, pet, response_style` (một giá trị), `interests, hobbies` (nhiều giá trị). Location dùng gazetteer địa danh, profession dùng pattern `<X> engineer/manager/...`. | Câu trả lời ngắn và chính xác theo từng field. Prompt view gọn hơn văn bản thô. | Gazetteer và pattern bị giới hạn: địa danh hay nghề lạ sẽ bị bỏ sót. |
| **Memory decay** | Interests xếp theo `mentions × 0.9^(số revision kể từ lần nhắc cuối)`, chỉ top-4 vào prompt. | Chủ đề nhắc nhiều và gần đây (Python, AI ứng dụng) được ưu tiên hơn chủ đề nhắc một lần đã lâu (async Python). Prompt không phình theo số topic. | Chủ đề quan trọng nhưng ít khi nhắc có thể bị đẩy ra khỏi prompt, dù vẫn còn trong file. |

Kiểm tra trên dataset này: hạ ngưỡng xuống 0.1 không làm đổi kết quả benchmark. Lý do là trong data, mỗi câu nhiễu đều đi kèm một câu khẳng định lại ngay sau đó, và extractor giữ candidate cuối cùng của mỗi message. Hiệu quả của ngưỡng được kiểm chứng riêng bằng test `test_extraction_handles_corrections_noise_and_questions`.

## 5. Test

`pytest src/test_agents.py -v` có 6 test, tất cả đều pass:

- `test_user_markdown_read_write_edit`: kiểm tra read/write/edit/file_size và việc sanitize user id thành path.
- `test_user_markdown_conflict_keeps_only_newest_fact`: correction mới thay fact cũ, fact cũ chỉ còn trong audit log và không vào prompt.
- `test_extraction_handles_corrections_noise_and_questions`: kiểm tra correction, câu đùa, nơi đi họp, và câu hỏi không bị ghi.
- `test_compact_trigger`: compact kích hoạt trên thread dài, message giữ lại ≤ keep, summary chứa nội dung cũ; hội thoại ngắn không compact.
- `test_cross_session_recall`: Advanced nhớ qua thread mới và qua cả agent instance mới (đọc lại từ đĩa). Baseline nhớ trong thread nhưng quên ở thread mới.
- `test_compact_reduces_prompt_load_on_long_thread`: tổng prompt của Advanced < 75% Baseline, và prompt của lượt cuối < 50% Baseline.

## 6. Tóm tắt

1. Baseline không nhớ dài hạn: recall bằng 0%.
2. Advanced thêm `User.md` nên recall lên 100%, kể cả với correction và nhiễu.
3. Hội thoại dài làm prompt cost của Baseline tăng theo O(n²): 2,469 token chỉ riêng ở lượt 16.
4. Compact memory chặn prompt mỗi lượt ở khoảng 500–850 token, giảm 50% tổng prompt tokens.
5. Đổi lại hệ thống phức tạp hơn: hội thoại ngắn tốn hơn, memory file tăng dần, và cần guardrail (confidence, conflict, decay) để không lưu sai.
