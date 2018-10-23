"""
iterates over the million playlist data set and outputs info
about what is in there.
"""
import sys
import json
import re
import collections
import os
import pickle
import random

random.seed(181022)

VARIOUS_ARTISTS_URI = '0LyfQWJT6nXafLPZqxe9Of'

class Spotify_train:
    """
    generate training data set (a set of MPDs->train)
    """
    def __init__(self, train_fullpaths, trk_min_count, art_min_count, save_dir):
        self.num_playlists = 0
        self.playlist_tracks = list()
        self.playlist_artists = list()

        self.track2artist = dict()
        self.track_histogram = collections.Counter()
        self.artist_histogram = collections.Counter()
     
        for fullpath in train_fullpaths:
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice['playlists']:
                self.process_playlist(playlist)
  
        o_dict = collections.OrderedDict(self.track_histogram.most_common())
        total_trk_uri_list, trk_count_list, track_uri2id = self.create_uri2id(o_dict, trk_min_count, 0)
        del o_dict
        
        del(self.artist_histogram[VARIOUS_ARTISTS_URI])
        
        o_dict = collections.OrderedDict(self.artist_histogram.most_common())
        total_art_uri_list, art_count_list, artist_uri2id = self.create_uri2id(o_dict,art_min_count,len(track_uri2id))
        del o_dict

        playlists = []
        print("len %d %d" % (len(self.playlist_tracks), len(self.playlist_artists)))
        for tracks_uri, artists_uri, title in zip(self.playlist_tracks, self.playlist_artists):
            tracks_id = self.change_uri2id(tracks_uri, track_uri2id)
            artists_id = self.change_uri2id(artists_uri, artist_uri2id)
            if len(tracks_id) == 0 and len(artists_id) == 0:
                continue
            if len(tracks_id) > 250 or len(artists_id) > 250:
                continue

            playlists.append([tracks_id, artists_id])
            self.num_playlists += 1

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
           
        file_data = dict()

        file_data['track_total'] = total_trk_uri_list
        file_data['track_count'] = trk_count_list
        
        file_data['track_uri2id'] = track_uri2id
        file_data['artist_uri2id'] = artist_uri2id
        
        file_data['playlists'] = playlists

        print('train')
        with open(save_dir+'/'+'train', 'w') as make_file:
            json.dump(file_data, make_file, indent="\t")
        print("num playlists: %d, tracks_total: %d, tracks>=min_count: %d, artists>=min_count: %d" %
              (self.num_playlists, len(total_trk_uri_list), len(track_uri2id), len(artist_uri2id)))
                
    def process_playlist(self, playlist):
        tracks = []
        artists = []
        for track in playlist['tracks']:
            trk_uri = track['track_uri'].split(':')[2]
            self.track_histogram[trk_uri] += 1
            tracks.append(trk_uri)
            
            art_uri = track['artist_uri'].split(':')[2]
            self.artist_histogram[art_uri] += 1
            artists.append(art_uri)
            if trk_uri not in self.track2artist:
                self.track2artist[trk_uri] = art_uri
            
        self.playlist_tracks.append(tracks)
        self.playlist_artists.append(artists)
    
    def create_uri2id(self, o_dict, min_count, start_from):
        uri_list = list(o_dict.keys())
        valid_uri_list = uri_list[:]
        count_list = list(o_dict.values())
        if min_count > 1:
            rm_from = count_list.index(min_count-1)
            del count_list[rm_from:]
            del valid_uri_list[rm_from:]
        uri2id = dict(zip(valid_uri_list, range(start_from, start_from+len(valid_uri_list))))
        return uri_list, count_list, uri2id
    
    def change_uri2id(self, uris, uri2id):
        ids = []
        for cur_uri in uris:
            cur_id = uri2id.get(cur_uri, -1)
            if cur_id == -1:
                continue
            ids.append(cur_id)
        return ids


class Spotify_test:
    """
    generate test data set (a set of MPDs->test)
    """
    def __init__(self, test_fullpaths, train_json, test_seeds_num, save_dir, is_shuffle):
        with open(train_json) as data_file:
            train = json.load(data_file)
        self.track_uri2id = train['track_uri2id']
        self.artist_uri2id = train['artist_uri2id']
        self.track_total = set(train['track_total'])
        self.is_shuffle = is_shuffle

        self.test_seeds_num = test_seeds_num
        self.num_playlists = 0
        self.playlists = list()

        for fullpath in test_fullpaths:
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice['playlists']:
                self.process_playlist_for_test(playlist)
           
        file_data = {}
        file_data['playlists'] = self.playlists

        name = 'test-'+str(test_seeds_num)
        if self.is_shuffle:
            name = name + 'r'
        print(name)
        with open(save_dir+'/'+name, 'w') as make_file:
            json.dump(file_data, make_file, indent="\t")
        print("num_playlists:%d" % self.num_playlists)
                
    def process_playlist_for_test(self, playlist):
        tracks = []
        artists = []
        for track in playlist['tracks']:
            track_uri = track['track_uri'].split(':')[2]
            artist_uri = track['artist_uri'].split(':')[2]
            # not consider tracks that did not appear in the training set.
            if track_uri not in self.track_total:
                continue

            track_id = self.track_uri2id.get(track_uri, -1)
            tracks.append(track_id)
            artist_id = self.artist_uri2id.get(artist_uri, -1)
            artists.append(artist_id)

        if len(tracks) <= self.test_seeds_num:
            return
        l_answers = len(tracks) - self.test_seeds_num
        if self.test_seeds_num == 0 and (l_answers < 10 or l_answers > 50):
            return
        elif self.test_seeds_num == 1 and (l_answers < 9 or l_answers > 77):
            return
        elif self.test_seeds_num == 5 and (l_answers < 5 or l_answers > 95):
            return
        elif self.test_seeds_num == 10 and (l_answers < 30 or l_answers > 90):
            return
        elif self.test_seeds_num == 25 and (l_answers < 76):
            return
        elif self.test_seeds_num == 100 and (l_answers < 50):
            return

        if self.is_shuffle:
            tracks_shuf = []
            artists_shuf = []
            index_shuf = list(range(len(tracks)))
            random.shuffle(index_shuf)
            for i in index_shuf:
                tracks_shuf.append(tracks[i])
                artists_shuf.append(artists[i])
            tracks = tracks_shuf
            artists = artists_shuf

        seeds_tracks = []
        seeds_artists = []
        answers = []

        for track, artist in zip(tracks[:self.test_seeds_num], artists[:self.test_seeds_num]):
            if track != -1:
                seeds_tracks.append(track)
            if artist != -1:
                seeds_artists.append(artist)

        for track in tracks[self.test_seeds_num:]:
            if track not in seeds_tracks:
                answers.append(track)

        self.num_playlists += 1
        self.playlists.append([seeds_tracks, seeds_artists, answers])