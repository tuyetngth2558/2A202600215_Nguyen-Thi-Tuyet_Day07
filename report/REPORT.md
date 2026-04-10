# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thị Tuyết
**Nhóm:** B1-C401
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai câu có vector embedding gần cùng hướng, tức là nội dung/ngữ nghĩa gần nhau dù từ ngữ có thể khác nhau. Giá trị càng gần 1 thì mức tương đồng ngữ nghĩa càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: Triết học nghiên cứu những quy luật chung nhất của tự nhiên, xã hội và tư duy.
- Sentence B: Triết học tập trung vào các nguyên lý phổ quát của thế giới và nhận thức.
- Tại sao tương đồng: Cả hai đều mô tả cùng bản chất của triết học bằng từ đồng nghĩa.

**Ví dụ LOW similarity:**
- Sentence A: Lịch sử triết học phương Tây cổ đại rất đa dạng.
- Sentence B: Hướng dẫn nấu bún bò Huế cần chuẩn bị xương bò và sả.
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (triết học vs nấu ăn), không chia sẻ ngữ cảnh chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng vector (ngữ nghĩa) thay vì độ dài vector, nên ổn định hơn khi biểu diễn văn bản có độ dài khác nhau. Với embeddings, hướng thường quan trọng hơn khoảng cách hình học tuyệt đối.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`  
> `= ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> *Đáp án:* `23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap = 100: `ceil((10000 - 100)/(500 - 100)) = ceil(9900/400) = 25 chunks`, tức tăng từ 23 lên 25. Overlap lớn hơn giúp giữ mạch ngữ nghĩa ở biên chunk, giảm mất ngữ cảnh khi retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Giáo trình Triết học Mác - Lê Nin

**Tại sao nhóm chọn domain này?**
> Đây là môn học bắt buộc với toàn bộ sinh viên đại học tại Việt Nam, nhưng tài liệu thường dày và trừu tượng, khiến sinh viên khó tra cứu nhanh khi ôn thi. Một RAG chatbot trên domain này cho phép sinh viên đặt câu hỏi tự nhiên như *"Vật chất là gì theo Lenin?"* và nhận câu trả lời trích dẫn đúng chương, đúng nguồn — thay vì phải lật từng trang giáo trình. Ngoài ra, nội dung giáo trình có tính ổn định cao (ít thay đổi theo năm), rất phù hợp để xây dựng và đánh giá một hệ thống RAG mà không lo dữ liệu bị lỗi thời.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Giáo trình Triết học Mác - Lê Nin | Đại học quốc gia Hà Nội - Trung tâm thư viện và trí thức số | ~683,585 | source="Giáo trình Triết học Mác-Lênin", level="Đại học", audience="Khối ngành ngoài lý luận chính trị" |
| 2 | 20K-AI-Handbook_final.pdf | Tài liệu chương trình AI nội bộ (PDF) | ~27,000 (sau trích xuất) | source="20K-AI-Handbook_final.pdf", category="faq", lang="vi", doc_type="pdf" |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | `Giao trinh Triet hoc.md` | Dùng để filter theo đúng tài liệu khi hỏi đáp, tránh trộn domain. |
| `topic` / `category` | string | `triet_hoc`, `faq`, `rag` | Thu hẹp không gian tìm kiếm theo chủ đề, tăng precision. |
| `lang` | string | `vi`, `en` | Hỗ trợ truy vấn đa ngôn ngữ, tránh nhầm ngữ cảnh giữa tiếng Việt/Anh. |
| `level` / `audience` | string | `Đại học`, `ngoài lý luận chính trị` | Phù hợp khi cần trả lời theo cấp độ và đối tượng học viên. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `Giao trinh Triet hoc.md` | FixedSizeChunker (`fixed_size`) | 855 | 799.5 | Trung bình (dễ cắt ngang ý ở cuối chunk) |
| `Giao trinh Triet hoc.md` | SentenceChunker (`by_sentences`) | 2059 | 327.4 | Cao (giữ ranh giới câu tốt) |
| `Giao trinh Triet hoc.md` | RecursiveChunker (`recursive`) | 1150 | 591.9 | Tốt (cân bằng độ dài và ngữ cảnh) |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> `RecursiveChunker` với thứ tự separator ưu tiên: xuống dòng đôi, xuống dòng đơn, dấu chấm, khoảng trắng, rồi mới cắt cứng theo `chunk_size`. Cách này giúp chunk bám cấu trúc tự nhiên của tài liệu học thuật (mục, đoạn, câu) trước khi phải cắt kỹ thuật. Với tài liệu dài như giáo trình, đặt `chunk_size` trung bình để tránh chunk quá ngắn gây nhiễu retrieval. Mỗi chunk được gắn metadata `doc_id`, `chunk_index`, `source` để dễ truy vết và lọc.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain triết học có đoạn văn dài, nhiều định nghĩa và luận điểm liên kết theo đoạn; nếu cắt cứng dễ vỡ ngữ nghĩa. `RecursiveChunker` tận dụng cấu trúc văn bản nên giữ được ý trọn vẹn hơn so với fixed-size thuần túy. Đồng thời số chunk không quá lớn như sentence-only, giúp cân bằng chất lượng và tốc độ truy xuất.

**Code snippet (nếu custom):**
```python
from src.chunking import RecursiveChunker

chunker = RecursiveChunker(chunk_size=1200)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `Giao trinh Triet hoc.md` | best baseline: `RecursiveChunker` (chunk_size=800) | 1150 | 591.9 | Tốt: giữ ngữ cảnh khá tốt nhưng còn nhiều chunk nhỏ gây nhiễu nhẹ |
| `Giao trinh Triet hoc.md` | **của tôi**: `RecursiveChunker` (chunk_size=1200) | 759 | 897.9 | Tốt hơn: giảm phân mảnh, chunk đầy đủ ý hơn, top-k ổn định hơn cho câu hỏi định nghĩa |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Thị Tuyết | RecursiveChunker | 8.0 | Giữ ngữ cảnh tốt, ít cắt ngang đoạn | Cần tinh chỉnh thêm theo chương |
| Nguyễn Mai Phương | ParentChildChunker (Small-to-Big) | 8 | Child nhỏ (319 chars) match chính xác thuật ngữ; parent lớn giữ ngữ cảnh section cho LLM; 4/5 queries relevant top-3 | Parent quá lớn (avg 26K chars) có thể vượt context window LLM; heading regex chỉ hoạt động tốt với giáo trình có format chuẩn |
| Chu Thị Ngọc Huyền | Sentence Chunking | 8 | Bảo toàn ngữ cảnh logic của lập luận triết học bằng cách tôn trọng ranh giới câu, giúp RAG retrieval cao hơn | Chunk size nhỏ hơn (422 vs 500 chars) có thể bỏ lỡ context nếu lập luận triết học kéo dài trên nhiều câu |
| Hứa Quang Linh | AgenticChunker | 9 | Tự phát hiện ranh giới chủ đề bằng embedding; mỗi chunk mang đủ ngữ cảnh 1 khái niệm triết học | Chunk lớn (avg ~4K chars) có thể chiếm nhiều context window; chạy chậm hơn (~97s trên 684K chars) |
| Chu Bá Tuấn Anh | RecursiveChunker | 8.5 | Cân bằng tốt giữa ngữ nghĩa và độ dài; giữ được cấu trúc tài liệu (paragraph/sentence); retrieval ổn định | Phụ thuộc heuristic nên đôi khi split chưa tối ưu; có thể tạo chunk rời rạc; cần tuning chunk size và overlap thêm |
| Nguyễn Văn Lĩnh | SentenceChunker(3) | 8.5 | Giữ ngữ pháp, retrieval scores cao (0.3-0.5) | Tăng số lượng chunk (1610 vs 300), có thể chậm retrieval |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với domain giáo trình triết học, `AgenticChunker` hiện cho kết quả tốt nhất về chất lượng retrieval (9/10) vì có khả năng tách theo ranh giới chủ đề bằng embedding, giúp mỗi chunk mang trọn một khái niệm triết học. Tuy nhiên, strategy này có chi phí cao hơn (chunk lớn, runtime chậm), nên khi triển khai thực tế cần cân nhắc tài nguyên và context window. Nếu ưu tiên cân bằng giữa chất lượng và hiệu năng, `RecursiveChunker`/`SentenceChunker` là lựa chọn an toàn hơn; còn nếu ưu tiên độ chính xác cao cho bài toán học thuật sâu, `AgenticChunker` là phương án nổi bật nhất.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex để tách câu theo dấu kết thúc phổ biến (`.`, `!`, `?`) và cả xuống dòng, sau đó loại bỏ khoảng trắng dư. Các câu được gom theo `max_sentences_per_chunk`. Edge case đã xử lý: text rỗng, nhiều khoảng trắng, hoặc câu không kết thúc bằng dấu câu chuẩn.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán thử tách theo thứ tự separator ưu tiên (`\\n\\n`, `\\n`, `. `, ` `, rồi cắt cứng). Nếu đoạn còn dài hơn `chunk_size` thì đệ quy với separator kế tiếp cho đến khi đạt kích thước phù hợp. Base case là đoạn rỗng, hoặc đoạn đã `<= chunk_size`, hoặc không còn separator để tách tiếp.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi `Document` được embed thành vector và lưu record chuẩn hóa gồm `id`, `content`, `metadata`, `embedding`. Khi search, query cũng được embed rồi tính điểm bằng dot product với từng record. Kết quả được sắp xếp giảm dần theo score và cắt `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` lọc trước theo `metadata_filter`, sau đó mới tính similarity trên tập đã lọc để tăng precision. `delete_document` xóa tất cả records có `metadata['doc_id'] == doc_id`, trả về `True` nếu có phần tử bị xóa. Cách này giúp xóa theo tài liệu gốc dù đã chunk thành nhiều phần.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent retrieve `top_k` chunks liên quan từ store, đánh số từng chunk rồi ghép thành phần `Context` trong prompt. Prompt yêu cầu LLM chỉ trả lời dựa trên context và nói không biết nếu thiếu thông tin. Cách này giúp câu trả lời grounded hơn thay vì sinh tự do.

### Test Results

```
# Paste output of: pytest tests/ -v
============================= test session starts =============================
collected 42 items
...
============================= 42 passed in 0.17s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Triết học nghiên cứu những quy luật chung nhất của tự nhiên, xã hội và tư duy. | Triết học tập trung vào các nguyên lý phổ quát của thế giới và nhận thức. | high | 0.0945 | Đúng (tương đối) |
| 2 | Con người có thể nhận thức thế giới khách quan thông qua thực tiễn. | Thực tiễn là cơ sở để kiểm nghiệm tính đúng sai của tri thức. | high | 0.1230 | Đúng (tương đối) |
| 3 | Phép biện chứng duy vật nhấn mạnh sự vận động và phát triển. | Kinh tế vi mô phân tích hành vi của doanh nghiệp trên thị trường. | low | -0.0639 | Đúng |
| 4 | Nội dung chương nói về mối quan hệ giữa vật chất và ý thức. | Chương này bàn về phạm trù vật chất, ý thức và vai trò của thực tiễn. | high | -0.1919 | Sai |
| 5 | Lịch sử triết học phương Tây cổ đại rất đa dạng. | Hướng dẫn nấu bún bò Huế cần chuẩn bị xương bò và sả. | low | 0.1420 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp số 4 gây bất ngờ nhất vì hai câu rất gần nghĩa nhưng score lại âm. Điều này cho thấy với `mock embedding` (deterministic hash), điểm số không phản ánh ngữ nghĩa thật như model embedding chuyên dụng. Vì vậy khi đánh giá similarity cần lưu ý backend embedding đang dùng.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Triết học là gì? | Triết học là hệ thống tri thức lý luận chung nhất về thế giới và vị trí con người trong thế giới đó. |
| 2 | Vấn đề cơ bản của triết học gồm những mặt nào? | Gồm mặt bản thể luận (vật chất - ý thức cái nào có trước) và mặt nhận thức luận (con người có khả năng nhận thức thế giới hay không). |
| 3 | Vai trò của thực tiễn đối với nhận thức là gì? | Thực tiễn là cơ sở, động lực, mục đích và tiêu chuẩn kiểm tra chân lý của nhận thức. |
| 4 | Phép biện chứng duy vật nhấn mạnh điều gì? | Nhấn mạnh sự vận động, phát triển và mối liên hệ phổ biến của sự vật hiện tượng. |
| 5 | Sự khác nhau giữa chủ nghĩa duy vật và duy tâm là gì? | Duy vật coi vật chất có trước, quyết định ý thức; duy tâm coi ý thức/tinh thần có trước. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Triết học là gì? | Đoạn mô tả quan niệm duy tâm về ý thức (không đúng trọng tâm định nghĩa triết học). | 0.4024 | Không | Trả lời theo context lấy về, thiếu định nghĩa chuẩn. |
| 2 | Vấn đề cơ bản của triết học gồm những mặt nào? | Đoạn nêu các nội dung chính khi định nghĩa triết học. | 0.3230 | Có (một phần) | Có đề cập đúng hướng nhưng chưa tách rõ 2 mặt. |
| 3 | Vai trò của thực tiễn đối với nhận thức là gì? | Đoạn nói về không gian-thời gian, liên quan yếu tới vai trò thực tiễn. | 0.3137 | Không | Câu trả lời chưa bám đúng trọng tâm. |
| 4 | Phép biện chứng duy vật nhấn mạnh điều gì? | Đoạn bàn về ý thức và nhận thức (liên quan gián tiếp). | 0.3766 | Một phần | Trả lời chưa nêu rõ tính vận động-phát triển. |
| 5 | Sự khác nhau giữa chủ nghĩa duy vật và duy tâm là gì? | Đoạn về vô thức (không đúng câu hỏi). | 0.3698 | Không | Trả lời thiếu chính xác do retrieval sai. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Dùng `ParentChildChunker (Small-to-Big)` (8.0/10) học được cách kết hợp `child chunk` nhỏ để match từ khóa chính xác và `parent chunk` lớn để giữ ngữ cảnh cho bước trả lời. Dùng `SentenceChunking` (8.0/10) giúp bảo toàn mạch lập luận triết học khi tôn trọng ranh giới câu. Dùng `AgenticChunker` (9.0/10) chunk theo chủ đề bằng embedding có thể tăng chất lượng retrieval, nhưng cần cân nhắc chi phí thời gian và context window.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua demo, nhóm có kết quả tốt thường không chỉ đổi chunker mà còn đổi cả chiến lược dữ liệu: chuẩn hóa heading, gắn metadata theo chương/mục, và kiểm soát độ dài chunk theo loại câu hỏi. Với tài liệu học thuật dài, việc thiết kế pipeline ingest quan trọng không kém model embedding. Bài học lớn nhất là phải tối ưu đồng thời `chunking + metadata + retrieval`, không tối ưu riêng lẻ từng phần.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Vẫn chọn `RecursiveChunker` làm nền (vì cân bằng tốt), nhưng sẽ thêm lớp parent-child cho các mục định nghĩa quan trọng để tăng độ chính xác khi hỏi khái niệm, bổ sung metadata bắt buộc gồm `chapter`, `section`, `keyword`, `concept_type` để filter trước khi xếp hạng sau đó chuyển sang local embedder và benchmark định kỳ trên 5 query cố định để theo dõi cải thiện theo từng lần tuning.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **85 / 100** |
