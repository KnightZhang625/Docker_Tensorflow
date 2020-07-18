docker run -p 8501:8501 \
  --mount type=bind,\
source="$(pwd)/application/toy_model/app_models",\
target="/models/toyModel" \
  -e MODEL_NAME="toyModel" -t tensorflow/serving &