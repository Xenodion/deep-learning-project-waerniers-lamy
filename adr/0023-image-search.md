# 0023. The visitor can search the collection with a picture, and the app takes uploads again, within limits

- **Status:** Accepted, revises the "no upload" rule of `deploy/README.md`
- **Date:** 2026-10-30

**Context.** Tab 2 of the demo finds paintings from a sentence: the SigLIP text tower against the SigLIP image embeddings
of the 600 Museum B paintings (`siglip_embeddings.npy`, ADR 0013, 9998). A visitor with a photo of a painting, or a
picture of a scene they like, has no sentence to type. The same embeddings answer that question with the SigLIP image
tower, no new model and nothing to train. Until now the app refused every upload (`max_file_size=1` byte): the server
only decoded images it shipped. Taking a picture means decoding files from anyone, on 2 GiB of memory of which about
1 GiB is the models.

**Decision.** We add a picture search under the text search of tab 2, and reopen uploads with two limits:

- **The picture is encoded exactly like the collection:** RGB, the SigLIP processor, L2 normalisation, then the same
  cosine ranking as the text search (one `rank()` for both). Encoding a painting of the collection gives back its stored
  vector (cosine 1.000 on 8 paintings) and the painting comes first. A picture is shrunk to 1024 px first, since SigLIP
  reads 224 px.
- **10 MB per file** (`max_file_size`): enough for a 48 Mpx phone photo. Gradio's upload route answers 413 above, before
  any decoding.
- **50 Mpx per picture** (`Image.MAX_IMAGE_PIXELS = 25_000_000`, Pillow refuses above twice that): a compressed file of
  200 KB can decode to gigabytes. The limit is global, so it also holds when Gradio opens the file, before our code.
- **Uploaded files leave Gradio's cache after 5 minutes** (`delete_cache`): on Cloud Run the disk is memory.
- **Same number of results as the text search, 6**, in the same grid; a text search clears the picture and the other way
  round, so the results always answer what is on screen.

Scored on Museum B: for each painting, **88.0%** of its 5 nearest paintings (itself left out) share its genre, against
**19.9%** by chance (bootstrap 95% interval 86.2% to 89.7%). By genre, from 75.5% (cityscape, confused with landscape)
to 95.5% (still life). `build_app.py` computes it and the Info tab shows it.

**Rejected.**
- *A list of allowed formats (JPEG, PNG, WebP).* `gr.Image` has no `file_types`, and Gradio opens the file before our
  code, so a check there comes too late. Filtering for real means a `gr.File` and our own decoding and preview, or
  removing formats from Pillow's registry (internal, may change between versions). The risk is memory, and the two
  limits above cover every format.
- *Our own error message for unreadable files.* A truncated file, a text renamed `.jpg` or a picture above 50 Mpx are
  all refused while Gradio decodes, before our handler: the visitor sees Gradio's generic error. `show_error=True` would
  show the message, but also internal paths.
- *Camera capture in the widget.* On a phone, the upload button already offers the camera.
- *A ranking metric with graded answers, like the 40 questions of the text search.* There is no visitor question behind
  a picture; genre agreement is a proxy for what the visitor wanted, measured on labels the museum already has.

**Consequences.**
- The visitor can search without words, the brief's point 2 in another form; the cost is a few lines and no new model.
- The public surface grows: the server decodes files from anyone. Tested refused: 11 MB (413), 64 Mpx (decompression
  bomb), truncated JPEG, text renamed `.jpg`. Tested accepted: JPEG, WebP, PNG with transparency, grey PNG, 896 px JPEG.
- A picture search took 65 to 850 ms on a laptop (the largest file slowest); the queue limits (2 jobs at a time, 16
  waiting) still bound the load.
- iPhone HEIC is not read by Pillow; Safari usually sends a JPEG instead. Revisit with `pillow-heif` if visitors hit it.
- 88% is about genre only: two landscapes of different moods both count as right. It says the search finds paintings of
  the same kind, not that a visitor would pick them.
