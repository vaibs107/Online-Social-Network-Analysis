from collections import Counter, defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import configparser
import json
import copy
import math
import urllib.request

consumer_key = 'tuEAiKfxj8Rj0vtId1QeFf7cl'
consumer_secret = 'dYnyDVDmBviI7iN0ai0CQfxsedvChkWajzj1NL2CPIEeD8UAsy'
access_token = '2199508622-YwB72doU0nKHJSv7GWKHAIc33JX2qreBkTBSO6M'
access_token_secret = '0xNBi1q6T4mmss2fwz9PpCNieMwXwq3ccyuaBg0qNhZzk'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def read_screen_names(filename):
    fptr = open(filename, "r")
    screen_names = []
    for names in fptr:
        screen_names.append(names.rstrip())
    return screen_names
    pass

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter, screen_names):
    users = []
    for sn in screen_names:
        users.append([user for user in robust_request(twitter,'users/lookup',{'screen_name':sn})][0])
    return users
    pass

def get_friends(twitter, screen_names):
    twit_ids = [u for u in robust_request(twitter,'friends/ids',{'screen_name':screen_names,'count':2500})]
    return sorted(twit_ids)
    pass

def add_all_friends(twitter, users):
    for user in users:
        user['friends'] = get_friends(twitter, user['screen_name'])
    json.dump(users, open('users.json', 'w', encoding='utf-8'))
    pass

def print_num_friends(users):
    candidates = sorted([(user['screen_name'],len(user['friends'])) for user in users])
    for candidate in candidates:
        print (candidate[0],candidate[1])
    pass

def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('teams.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)

if __name__ == '__main__':
    main()