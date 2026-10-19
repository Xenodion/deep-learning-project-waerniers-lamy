# FINAL PROJECT GUIDELINES

You are a member of the AI team of a (fictional) mid-sized French fine-arts museum (the Museum below). The
Museum wants an intelligent guide, available on visitors' phones and on kiosks in the galleries, that can do two
things

1.  Recognise what the visitor is looking at. The visitor points the camera at a painting; the guide recognises its
style (Impressionism, Baroque, Cubism…) or its genre (portrait, landscape, still life…) and tells the story of
that movement or tradition.

2.  Answer questions in natural language. A visitor types "Which painting shows a stormy sea?" or "I would like
to see portraits of children"; the guide proposes the artworks of the collection that best match the question.

The director also expects a clear recommendation: the kiosks run on modest CPUs, the team is small and the
budget is limited. Is a neural network worth it in production, or would a simpler classical model do the job?
Your mission is to build, diagnose and defend the neural-network components of this guide, in the spirit of the
course: networks built from PyTorch primitives rather than library magic, every number measured, every claim
backed by evidence. The project is mutualised: the scenario and the assessment format (Git deliverable, group oral
defence, individual deliberation, peer review) are shared at programme level

## DATA
Can be loaded from the attached Python File

## DELIVERABLE
- A feed-forward network in PyTorch for artwork classification
- Training monitoring and failure diagnosis
- A pretrained CNN (ResNet) for image classification
- Deadline is for the 30th of October 2026

The Source Code must be delivered with Clear Git History on Github or Gitlab.

You will also need to prepare a 10 minutes PowerPoint Presentation followed by a 5 Minutes Q&A.
