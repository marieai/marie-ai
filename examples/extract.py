# Import necessary python libraries
import os
import requests
import time
import mimetypes

# Specify variables for use in script below
api_base_url = 'http://127.0.0.1:5000/api'
default_queue_id = '0000-0000-0000-0000'

# Use the API key to define headers for authorization
api_key = 'MY_API_KEY'

auth_headers = {
    'Authorization': f'Bearer {api_key}'
}

def process_extract(queue_id: str, file_location: str) -> str:
    if not os.path.exists(file_location):
        raise Exception(f'File not found : {file_location}')

    upload_url = f'{api_base_url}/extract/{queue_id}'

    # Prepare file for upload
    with open(file_location, 'rb') as file:
        mime_type = mimetypes.guess_type(file_location)[0]

        files_to_upload = [
            ('file', (file_location, file, mime_type))
        ]

        # Upload file to api
        print(f'Uploading {file} to marie-icr for processing')
        upload_json = requests.post(
            upload_url,
            headers=auth_headers,
            files=files_to_upload
        ).json()

        print(upload_json)


if __name__ == '__main__':
    print('done')
    # Specify the path to the file you would like to process
    # file_location = './set-001/test/fragment-003.png'
    file_location = '/home/gbugaj/dev/sk-snippet-dump/hashed/AARON_JANES/38585416cc1f22103336f3419390c9ce.tif'
    process_extract(queue_id=default_queue_id, file_location=file_location)
