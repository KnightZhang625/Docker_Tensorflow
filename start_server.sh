docker run -p 8500:8500 \
  --mount type=bind,\
source="$(pwd)/application/toy_model/app_models",\
target="/models/toyModel" \
  -e MODEL_NAME="toyModel" -t tensorflow/serving &