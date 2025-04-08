aws s3 --no-sign-request sync s3://arxiv-tagging ./data/

unzip ./data/data.zip -d ./data/

rm ./data/data.zip