# import urllib.request as urllib2
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from unidecode import unidecode
import string
from tqdm import tqdm

quote_page = 'https://www.metrolyrics.com/{}-lyrics-{}.html'
input_filename = 'songs_list.csv'
output_filename = 'songs.csv'

songs_list = pd.read_csv(input_filename)
songs = songs_list.copy()
songs['Lyrics'] = songs['Lyrics'].astype('str')

print('Saving songs listed in songs_list.csv')
for index, row in tqdm(songs_list.iterrows()):
    page = requests.get(quote_page.format(row['Song'], row['Artist']), timeout=30)
    soup = BeautifulSoup(page.text, 'html.parser')
    verses = soup.find_all('p', attrs={'class': 'verse'})

    if not verses:
        print('Could not save {} by {}'.format(row['Song'], row['Artist']))
        continue

    
    lyrics = ''
    for verse in verses:
        text = verse.text.strip()
        text = re.sub(r"\[.*\]\n", "", unidecode(text))
        text = re.sub(r"\([^)]*\)", '', unidecode(text)) #removes parantheses and its contents
        text = text.translate(str.maketrans('', '', string.punctuation)) #removes punctuation
        
        if lyrics == '':
            lyrics = lyrics + text.replace('\n', '|-|')
        else:
            lyrics = lyrics + '|-|' + text.replace('\n', '|-|')
    
    songs.at[index, 'Lyrics'] = lyrics

    songs.head()

print('writing to .csv')
songs.to_csv(output_filename, sep=',', encoding='utf-8')