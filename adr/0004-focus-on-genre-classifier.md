# 0004. Focus on the genre classifier; leave the language feature for later

- **Status:** Accepted
- **Date:** 2026-10-20

**Context.** The brief describes two features (recognise a painting, answer questions in natural language), but the
listed deliverables only cover the vision side. Our teacher confirmed that either target is acceptable: the genre
(5 classes) or the style (27 classes). On the language feature he gave no guidance and it was barely covered in
class. The deadline is 30 October 2026.

**Decision.** We will put all our effort into the classifier that predicts the genre (5 classes) from the image: dense
baseline, monitoring and failure diagnosis, pretrained ResNet, all developed on museum A (train and validation) and
scored at the end on museum B ([0003](0003-simulated-museum-b-test-set.md)). We will not build the language feature in a second time.

**Rejected.** Building both features (it takes time from the graded work).

**Consequences.** We still need a classical baseline to answer the director's "is a neural network worth it?", trained
and scored on the same museum A / museum B split as the networks. If the
language part comes back, note that the dataset has no captions, only genre, style and artist.
