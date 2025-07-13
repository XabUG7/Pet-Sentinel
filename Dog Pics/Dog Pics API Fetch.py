import requests
import os
from time import sleep


output_dir = 'dog_images'

os.makedirs(output_dir, exist_ok=True)

url = 'https://dog.ceo/api/breeds/image/random/50'

total_images = 1000
batch_size = 50
num_batches = total_images // batch_size

img_counter = 1

for batch in range(num_batches):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_urls = response.json()["message"]

        for image_url in image_urls:
            img_data = requests.get(image_url).content
            filename = os.path.join(output_dir, f"dog_{img_counter:04d}_" + os.path.basename(image_url))
            with open(filename, "wb") as handler:
                handler.write(img_data)
            print(f"[{img_counter}/{total_images}] Saved: {filename}")
            img_counter += 1

        sleep(0.2)

    except Exception as e:
        print(f"Error on batch {batch+1}: {e}")
        continue
