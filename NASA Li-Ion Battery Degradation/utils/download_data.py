def download_data():
    if not os.path.isdir('data'):
        os.makedirs('data')
    if 'train_FD004.txt' not in os.listdir('data'):
        print('Downloading Data...')
        # Download the data
        r = requests.get("https://ti.arc.nasa.gov/c/6/", stream=True)
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall('data')
    else:
        print('Using previously downloaded data')