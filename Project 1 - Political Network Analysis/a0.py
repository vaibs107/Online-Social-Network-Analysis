"""
Online Social Network Analysis - Project 1
In this project, a list of Twitter accounts of 4 U.S. presedential candidates is taken.

The goal is to use the Twitter API to construct a social network of these
accounts. I have then used the [networkx](http://networkx.github.io/) library
to plot these links, as well as print some statistics of the resulting graph.

Steps:
1. Created an account on [twitter.com](http://twitter.com).
2. Generated authentication tokens by following the instructions on link ->(https://dev.twitter.com/docs/auth/tokens-devtwittercom).
3. Added the tokens to the key/token variables below. (API Key == Consumer Key)
4. Installed the below Python modules
[networkx](http://networkx.github.io/) and
[TwitterAPI](https://github.com/geduldig/TwitterAPI). 

The output of this analysis is provided in Log.txt.
"""

#Imports
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


# Credentials are put in the file twitter.cfg.
def get_twitter():
    """ Constructs an instance of TwitterAPI using the tokens entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    """
    Reads a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.
    """
    fptr = open(filename, "r")
    screen_names = []
    for names in fptr:
        screen_names.append(names.rstrip())
    return screen_names
    pass


# The method below is implemented to handle Twitter's rate limiting.
# This method is called whenever I need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    This function does this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
    """Retrieves the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    The API documentation can be found here: https://dev.twitter.com/rest/reference/get/users/lookup
    """
    users = []
    for sn in screen_names:
        users.append([user for user in robust_request(twitter,'users/lookup',{'screen_name':sn})][0])
    return users
    pass


def get_friends(twitter, screen_names):
    """ Returns a list of Twitter IDs for users that this person follows, up to 5000.
    Reference: https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.
    """
    twit_ids = [u for u in robust_request(twitter,'friends/ids',{'screen_name':screen_names,'count':5000})]
    return sorted(twit_ids)
    pass


def add_all_friends(twitter, users):
    """ Gets the list of accounts each user follows.
    I.e., calls the get_friends method for all 4 candidates.

    Stores the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing
    """
    for user in users:
        user['friends'] = get_friends(twitter, user['screen_name'])
    pass


def print_num_friends(users):
    """Prints the number of friends per candidate, sorted by candidate name.
    Log.txt shows an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    candidates = sorted([(user['screen_name'],len(user['friends'])) for user in users])
    for candidate in candidates:
        print (candidate[0],candidate[1])
    pass


def count_friends(users):
    """ Counts how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation can be found here: https://docs.python.org/dev/library/collections.html#collections.Counter
    """
    friend_list = []
    for user in users:
        friend_list.extend(user['friends'])
    return Counter(friend_list)
    pass


def friend_overlap(users):
    """
    Computes the number of shared accounts followed by each pair of users.

    Args:
        users...The list of user dicts.

    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list is
        sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order).
    """
    common_friends = []
    for i in range(len(users)):
        for j in range(i+1,len(users)):
            number_of_common_friends = set(users[i]['friends']) & set(users[j]['friends'])
            common_friends.append((users[i]['screen_name'],users[j]['screen_name'],len(number_of_common_friends)))
    return common_friends
    pass


def followed_by_hillary_and_donald(users, twitter):
    """
    Finds and returns the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. I have used the TwitterAPI to convert
    the Twitter ID to a screen_name. Reference: https://dev.twitter.com/rest/reference/get/users/lookup

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
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
    """ Creates a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates are added to the graph,
        only friends are added to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    graph = nx.Graph()
    
    for friend in friend_counts:
        if friend_counts[friend] > 1:
                for user in users:
                    if friend in user['friends']:
                        graph.add_edge(user['id'],friend)
    return graph
    
    pass


def draw_network(graph, users, filename):
    """
    Draws the network to a file. Only the candidate nodes are labelled; the friend
    nodes have no labels (to reduce clutter).

    Methods used here are: networkx.draw_networkx, plt.figure, and plt.savefig.
    """
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
