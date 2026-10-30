# Deploying the online demo

The demo is `demo.py` (Gradio) packaged with Museum B in a Docker image and served by Google Cloud Run. The image is
tagged by date (a tag must identify a build for good) and pushed to a private Artifact Registry.

```bash
# 1. the bundle: images, thumbnails, embeddings, weights and a copy of demo.py (checks that it reproduces 86.0%)
uv run --with gradio python scripts/build_app.py

# 2. the image
TAG=$(date +%F)
IMAGE=europe-west1-docker.pkg.dev/<project>/museum/museum-guide:$TAG
docker build -f deploy/Dockerfile -t $IMAGE app
docker push $IMAGE

# 3. the service
gcloud run deploy museum-guide --project <project> --region europe-west1 --image $IMAGE \
  --cpu 2 --memory 2Gi --min-instances 0 --max-instances 1 --concurrency 20 --timeout 120 \
  --cpu-boost --port 8080 --allow-unauthenticated
```

## What is exposed, and the limits

A public URL, so the surface is kept small:

- **One upload, the picture search of tab 2** ([ADR 0023](../adr/0023-image-search.md)). At most 10 MB per file
  (`max_file_size`: the upload route answers 413 before decoding) and 50 Mpx per picture (`Image.MAX_IMAGE_PIXELS`:
  Pillow refuses decompression bombs, also when Gradio opens the file). Any format Pillow reads is accepted; unreadable
  files get Gradio's generic error. Uploaded files are never served and leave Gradio's cache after 5 minutes (the disk
  of Cloud Run is memory). The image widget of tab 1 only displays.
- **One text field**, capped at 200 characters.
- **Files served:** only `images/` and `thumbs/` (`allowed_paths`); the code, weights and CSV answer 403.
- **Rate limits:** one instance at most (`--max-instances 1`), 20 concurrent requests, in the app 2 jobs at a time and
  16 waiting, 120 s per request. Cloud Run has no per-IP limit without a load balancer (Cloud Armor, paid), so a
  determined visitor can fill the queue but not the bill.
- **Budget alert** of 5 EUR on the project (50, 90, 100 %). It warns, it does not cut anything.
- **Container:** non-root user, no secret in the image, models downloaded at build time (no network at start-up),
  dependencies audited with `pip-audit` (none known at the time of writing).

Scale to zero means the first visit after a quiet period takes about 40 seconds. Cost at this volume: free tier.
