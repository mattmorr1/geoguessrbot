#!/usr/bin/env bash
# serve.py on Cloud Run, scale to zero (idle = bucket storage only). Usage: ./deploy.sh setup | upload <dir> | deploy | domain
# upload streams a serve bundle (data/{meta,feats,backbones}, runs/ laid out as in the repo) from matt-pc to the bucket.
set -euo pipefail
P=$(gcloud config get-value project 2>/dev/null) R=us-east1 S=alphabot DOMAIN=alphabot.mattmorr.com
B=gs://$P-$S SA=$S-run@$P.iam.gserviceaccount.com
case "${1:-deploy}" in
  setup)
    gcloud storage buckets create "$B" --location "$R" --uniform-bucket-level-access
    gcloud iam service-accounts create "$S-run" --display-name "$S Cloud Run"
    gcloud storage buckets add-iam-policy-binding "$B" --member "serviceAccount:$SA" --role roles/storage.objectViewer ;;
  upload)
    ssh matt-pc "cd ${2:?bundle dir on matt-pc} && find -L . -type f" | sed 's|^\./||' | while read -r f; do
      ssh -n matt-pc "cat '${2}/$f'" | gcloud storage cp - "$B/serve/$f"
    done ;;
  deploy)
    gcloud run deploy "$S" --source . --region "$R" --service-account "$SA" --allow-unauthenticated \
      --cpu 8 --memory 16Gi --min-instances 0 --max-instances 2 --concurrency 8 --timeout 300 --cpu-boost \
      --execution-environment gen2 --set-env-vars "BUNDLE=$B/serve" ;;
  domain)
    gcloud beta run domain-mappings create --service "$S" --domain "$DOMAIN" --region "$R" ;;
esac
