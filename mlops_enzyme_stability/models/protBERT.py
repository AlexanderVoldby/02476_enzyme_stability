from transformers import BertModel, BertTokenizer
import torch
from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
model = model.to(device)
model.eval()

credentials_dict = {
    'type': 'service_account',
    'client_id': os.environ["103753295588258345256"],
    'client_email': os.environ["vertex-service-account@enzyme-stability-02476.iam.gserviceaccount.com"],
    'private_key_id': os.environ["898e2071630a085122400e193e2424e6b992538c"],
    'private_key': os.environ["-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDWyWTde/BHcmP1\nXXRoHW+gHl+X5Te83k+4jsjkJCIgqeBcF39s9sArHYaDTj6QZD4FVqG17IYqX0WE\nbQ1Ayyr+OKTOygb9aCIW4ciUMQigecYCeRmGqSMAB0emIDQaZFpFhojvW9ixLExC\nmrMPdwhj+GujDCh6JTo4XNd+0wRgisAmI0ZCML//TepJhC2gNbbJSOtR/KFuJ0n4\nRUkf4lQmgA7bFx3Ilvu8aetPuhJ4dSkvjJNcjSbSxXE9FzHucyzZmNwYxchmmJWF\nxm1ctESaJmmnPeOp9FUNY+BlXkH9FJy7UojWtM2+XU31+nk8DZUj0kJ14G6mnwIW\n8brTREgbAgMBAAECggEAMzVW5/tpoQ8jx9cdLsl92JYFQoiyzkPOi/j26nW28USv\nKiWsCsWVsXHbpSidf/12T/2EloQqxKVsRJNfaohF2tCUcnz2R6pxHjV8heBe5yvB\nSsumi1V9877IlVHuOjCc2SCnOzIRqsQd9m89q708ceFDgF8gs2Z2yANSmXkj/QQH\nqVtfmQlIsOBdU43fO2eSeIDna9CqpoY/spAmN7amBFZ2Ih9pW9Aplo6yU3yy8uHp\nOq1t3ecv9Sx00zMNaF53HwFAnRkj/l0HGk6TdJ0dDP/9pUqjhdA2GQ5YhWabTJ+E\ne5biJaULJyWTTHgLaqbmwldJAm+xFV8sa6NQsaWtsQKBgQD35JCte28wqLcSCMTr\nLGVRW/h0hNeXPPPbwV03SCVd6oZzWwPSB/9x8dhqGyRhSniCX4k5A6v5xg0/j4Cp\nKyYui/yZTQhTtCkxRzn5zz6JMSrHBaE7XctfpGLrVYPf0y/CdF/mPPpPbS0E9TG6\nCAzGQoqRBQcbWklagKuL+odGkwKBgQDdz6d0ULYYeD22Y3YoXVJwJBJagsKIFW6B\n8aMx6JES0xfcZ3UqgWgdyuQyWKEOMs5db3OO2jr6nSM/RBwekj/Ho3CnKJNpulT0\n4HXpL8RQswFTioWwqY2bZGBNfZMaATcqBC1PgjKsYsXptoa4o1womEWwRAb2cvKN\nHRHUp0ClWQKBgCy8R8u28drzJ46OnJLgQSyMj2rfqlR1wIBRBfR7BZtMPpVEwIy/\nur6iaW7ElS0lllfYy9fJLNj3f96PlCVzTwGpa51yxGTup1xoQTMuzldN0y11e6JO\nC+ynqt5TYWgcIYtTGxdeu8Fnr28snJu45i1FRuJi6ORrx78YZ1zsiksXAoGBAKiZ\nichg/Tj2VLpJOewOm3YABS9lSyaTW89L8+cgxv0PFYkD3sxzVsemi/Q06B7ZwYDG\nEYlZGhGa1crmI5WdRvuhoSR+NCbeamtEHnwkQc2xcuWkWmBhUPD3yDe+pszSdbLr\nP+G6rnfYEGXIxvibu5ZjwDzuSHiWAQYAPahthTbpAoGAHgQohid6Byf2eBM9ip6q\nWBImPTkb41gGxBfHsjfSK1DxBV8TLqXAM7RwRrqqT8LspvrzayRHFKooj8xUgHvI\n5JmWyVLfT/zY9ybgWa1JttQccSmPcilV5NuqFcnXteCgujk8TXIVY8q2GTCSxyrV\nPlW61mR76ao8/AhcE3StpCM=\n-----END PRIVATE KEY-----\n"],
}
credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    credentials_dict
)
gcs_project = 'enzyme-stability-02476'
gcs_bucket = 'dtu-mlops-storage'
gcs_path = f'gs://{gcs_bucket}/{gcs_project}/models/protBERT/'
model.save_pretrained(gcs_path+"model")
tokenizer.save_pretrained(gcs_path+"tokenizer")

# client = storage.Client(credentials=credentials, project='enzyme-stability-02476')
# bucket = client.get_bucket('mybucket')
# blob = bucket.blob('../models/protBERT/model')
# blob.upload_from_filename('../models/protBERT/model')

# blob = bucket.blob('../models/protBERT/tokenizer')
# blob.upload_from_filename('../models/protBERT/tokenizer')
