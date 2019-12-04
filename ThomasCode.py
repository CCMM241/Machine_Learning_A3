#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:59:47 2019

@author: thomasdorveaux
"""

import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import re
import os
import zlib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
import seaborn as sns
from sklearn.cluster import KMeans

Spotify = pd.read_csv('SpotifyFeatures.csv')
lyrics = pd.read_csv('only_lyrics.csv')

#Merging lyrics and the songs

Spotify['track_name']=Spotify['track_name'].str.lower()
Spotify['track_name']=Spotify['track_name'].str.strip()
Spotify['artist_name']=Spotify['artist_name'].str.lower()
Spotify['artist_name']=Spotify['artist_name'].str.strip()
lyrics['song']=lyrics['song'].str.lower()
lyrics['song']=lyrics['song'].str.strip()
lyrics['artist']=lyrics['artist'].str.lower()
lyrics['artist']=lyrics['artist'].str.strip()

song_lyrics = Spotify.merge(lyrics, left_on=['track_name', 'artist_name'], right_on=['song', 'artist']) 
song_lyrics =song_lyrics.sort_values(by=['popularity'],ascending=False)
song_lyrics = song_lyrics.reset_index()
song_lyrics=song_lyrics.drop(['artist', 'song', 'link'], axis=1)

#preprocessing lyrics

def clean_lyrics(lyrics):
    new_lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
    new_lyrics = new_lyrics.replace("\n", "")
    new_lyrics = os.linesep.join([s for s in new_lyrics.splitlines() if s])
    return(new_lyrics)

def get_compression_rate(lyrics):
    original = lyrics.encode('utf-8')
    compressed = zlib.compress(original)
    decompressed = zlib.decompress(compressed)
    
    compression_rate = (len(original)-len(compressed))/len(original)
    return compression_rate

#Apply clean_lyrics function to text column
song_lyrics['text'] = song_lyrics['text'].map(clean_lyrics)

#Append new column with compression rate
song_lyrics['compression_rate'] = song_lyrics['text'].map(get_compression_rate)

#Drop duplicates
song_lyrics = song_lyrics.sort_values(by='popularity', ascending=False)
song_lyrics = song_lyrics.drop_duplicates(subset='track_id', keep="first")

#Addition of classes

quantile=song_lyrics['popularity'].quantile(np.arange(0, 1.01, 0.01).tolist())
quantile=quantile.reset_index()
quantile.rename(columns={'index':'quantile'}, inplace=True)
sns.lineplot(quantile['popularity'],quantile['quantile'])

kmeans = KMeans(n_clusters=5)
kmeans.fit(quantile)
quantile['cluster']=kmeans.predict(quantile)

plt.scatter(quantile['popularity'],quantile['cluster'])

one=quantile[quantile.cluster==1]
two=quantile[quantile.cluster==2]
three=quantile[quantile.cluster==3]
zero=quantile[quantile.cluster==0]
four=quantile[quantile.cluster==4]
min_one=np.min(one['popularity'])
max_one=np.max(one['popularity'])
min_two=np.min(two['popularity'])
max_two=np.max(two['popularity'])
min_three=np.min(three['popularity'])
max_three=np.max(three['popularity'])
min_zero=np.min(zero['popularity'])
max_zero=np.max(zero['popularity'])
min_four=np.min(four['popularity'])
max_four=np.max(four['popularity'])
print(1,min_one,max_one)
print(2,min_two,max_two)
print(3,min_three,max_three)
print(4,min_four,max_four)
print(0,min_zero,max_zero)

conditions = [
    (song_lyrics['popularity'] >= 78)&(song_lyrics['popularity'] <=100),
    (song_lyrics['popularity'] >= 55)&(song_lyrics['popularity'] <=77),
    (song_lyrics['popularity'] >= 39)&(song_lyrics['popularity'] <=54),
    (song_lyrics['popularity'] >= 19)&(song_lyrics['popularity'] <=38),
    (song_lyrics['popularity'] >= 18)&(song_lyrics['popularity'] <=0)]
choices = [4, 3, 2,1,0]
song_lyrics['label'] = np.select(conditions, choices)
song_lyrics.head()


#Display variables distribution 
f, axes = plt.subplots(2, 5, figsize=(14, 8), sharex=False)
sns.distplot(song_lyrics['popularity'],ax=axes[0,0])
sns.distplot(song_lyrics['acousticness'],ax=axes[0,1])
sns.distplot(song_lyrics['danceability'],ax=axes[0,2])
sns.distplot(song_lyrics['energy'],ax=axes[0,3])
sns.distplot(song_lyrics['instrumentalness'],ax=axes[0, 4])
sns.distplot(song_lyrics['liveness'],ax=axes[1,0])
sns.distplot(song_lyrics['loudness'],ax=axes[1,1])
sns.distplot(song_lyrics['speechiness'],ax=axes[1,2])
sns.distplot(song_lyrics['valence'],ax=axes[1,3])
sns.distplot(song_lyrics['tempo'],ax=axes[1,4])

#duration and discrete variables
sns.distplot(song_lyrics['duration_ms'])

sns.distplot(song_lyrics['key'])

sns.distplot(song_lyrics['mode'])
sns.distplot(song_lyrics['time_signature'])