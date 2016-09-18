# coding: utf-8

"""
Collecting a political social network - The goal is to use the Twitter API to construct a social network of the 4 US presidential candidates' accounts. 
"""

from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI

consumer_key = 'tuEAiKfxj8Rj0vtId1QeFf7cl'
consumer_secret = 'dYnyDVDmBviI7iN0ai0CQfxsedvChkWajzj1NL2CPIEeD8UAsy'
access_token = '2199508622-YwB72doU0nKHJSv7GWKHAIc33JX2qreBkTBSO6M'
access_token_secret = '0xNBi1q6T4mmss2fwz9PpCNieMwXwq3ccyuaBg0qNhZzk'

def get_twitter():
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
    twit_ids = [u for u in robust_request(twitter,'friends/ids',{'screen_name':screen_names,'count':5000})]
    return sorted(twit_ids)
    pass

def add_all_friends(twitter, users):
    for user in users:
        user['friends'] = get_friends(twitter, user['screen_name'])
    pass

def print_num_friends(users):
    candidates = sorted([(user['screen_name'],len(user['friends'])) for user in users])
    for candidate in candidates:
        print (candidate[0],candidate[1])
    pass

def count_friends(users):
    friend_list = []
    for user in users:
        friend_list.extend(user['friends'])
    return Counter(friend_list)
    pass

def friend_overlap(users):
    common_friends = []
    for i in range(len(users)):
        for j in range(i+1,len(users)):
            number_of_common_friends = set(users[i]['friends']) & set(users[j]['friends'])
            common_friends.append((users[i]['screen_name'],users[j]['screen_name'],len(number_of_common_friends)))
    return common_friends
    pass

def followed_by_hillary_and_donald(users, twitter):
    common_friend_hillary_and_donald = ""
    for i in range(len(users)):
        if (users[i]['screen_name'] == 'HillaryClinton'):
            for j in range(i+1,len(users)):
                if (users[j]['screen_name'] == 'realDonaldTrump'):
                    common_friend_id = set(users[i]['friends']) & set(users[j]['friends'])
    
    common_friend_hillary_and_donald = [user['screen_name'] for user in robust_request(twitter, 'users/lookup', {'user_id':common_friend_id}, max_tries=5)]
    return common_friend_hillary_and_donald
    pass

def create_graph(users, friend_counts):
    graph = nx.Graph()
    
    for friend in friend_counts:
        if friend_counts[friend] > 1:
                for user in users:
                    if friend in user['friends']:
                        graph.add_edge(user['id'],friend)
    return graph
    pass

def draw_network(graph, users, filename):
    candidate = {}
    for user in users:
        candidate[user['id']] = user['screen_name']
    plt.figure(figsize=(25,25))
    nx.draw_networkx(graph, nx.spring_layout(graph), labels = candidate, font_size=18, font_color='blue', font_weight="bold", node_color='black', edge_color='red', with_labels=True)
    plt.axis('off')
    plt.savefig(filename, format = "PNG")
    pass

def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()
