import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
import re
from unidecode import unidecode
import string
from tqdm import tqdm
import time

artist_songs_page = 'https://www.metrolyrics.com/{}-alpage-1.html'
input_filename = 'artist_list.csv'
output_filename = 'songs.csv'

artist_list = pd.read_csv(input_filename)

songs = pd.DataFrame(columns=['Artist', 'Song', 'Lyrics'])
songs['Artist'] = songs['Artist'].astype('str')
songs['Song'] = songs['Song'].astype('str')
songs['Lyrics'] = songs['Lyrics'].astype('str')

data = []

# get artist, songname, and link to song page
print('Getting song links for...') 
for index, row in artist_list.iterrows():
    print(row['Artist'])

    song_list_pages = []
    response = requests.get(artist_songs_page.format(row['Artist']), timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')
    soup = soup.find_all('span', attrs={'class': 'pages'})

    if not soup:
        song_list_pages.append(artist_songs_page.format(row['Artist']))
    else:
        content = soup[0].contents
        for item in content:
            if isinstance(item, Tag):
                song_list_pages.append(item['href'])
    
    for page in song_list_pages:
        response = requests.get(page, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        song_list = soup.find('table', attrs={'class': 'songs-table compact'})
        song_tags = song_list.find_all('a', attrs={'class': 'title'})
        for tag in song_tags:
            link = tag['href']
            song_name = re.search('.com/(.*)-lyrics', link).group(1)
            data.append([row['Artist'], song_name, link])

# get and clean lyrics from each song page
print('Extracting and cleaning scraped lyrics...')
for item in tqdm(data):
    artist, song_name, link = item
    page = requests.get(link, timeout=30)
    soup = BeautifulSoup(page.text, 'html.parser')
    verses = soup.find_all('p', attrs={'class': 'verse'})

    if not verses:
        item[-1] = ''
        continue

    lyrics = ''
    for verse in verses:
        text = verse.text.strip()
        text = re.sub(r"\[.*\]\n", "", unidecode(text))
        text = re.sub(r"\([^)]*\)", '', unidecode(text)) #removes parantheses and its contents
        text = text.translate(str.maketrans('', '', string.punctuation)) #removes punctuation
        text = text.lower() #make all lowercase
        
        if lyrics == '':
            lyrics = lyrics + text.replace('\n', ' |-| ')
        else:
            lyrics = lyrics + ' |-| ' + text.replace('\n', ' |-| ')
    
    item[-1] = lyrics

# add data to songs dataframe and shuffle
songs = songs.append(pd.DataFrame(data, columns=['Artist', 'Song', 'Lyrics']))
songs = songs[songs['Lyrics'].astype(bool)]
songs = songs.sample(frac=1)

# write to csv
print('writing to .csv')
songs.to_csv(output_filename, sep=',', encoding='utf-8')