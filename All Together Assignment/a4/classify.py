from TwitterAPI import TwitterAPI
from collect import robust_request, get_users, get_twitter
from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

def read_screen_names(filename):
    target_file = open (filename,'r')
    line = (target_file.read())
    return (line.split())
    target_file.close()

def gather_tweets(twitter, users):
# Get this person's timeline
    tweetdict ={}
    tweet_list = []
    for u in users:
        timeline = [tweet['text'] for tweet in twitter.request('statuses/user_timeline',{'screen_name': u['screen_name'],'include_rts':0,'count': 1000})]
        tweetdict[u['screen_name']]=timeline
    for key, value in tweetdict.items():
        with open('vaibhavtweets.txt', mode='a', encoding='utf-8') as myfile:
            for tweet1 in value:
                myfile.write(tweet1+'\n'+'\n')
    return(len(tweet_list))

def read_tweets(filename):
    target_file = open (filename,'r',encoding='UTF-8')
    line = (target_file.read())
    line.lower()
    return (line.split())
    target_file.close()

def afinn_sentiment(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    total = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    print("Positive sentiment count =",pos)
    print("Negative sentiment count =",neg)
    total = pos - neg
    print('total =',total)
    if(total>0):
        print("The overall sentiment of the document is positive")
    else:
        print("The overall sentiment of the document is negative")
    return pos,neg

def main():
    screen_names = read_screen_names('twitterdata.txt')
    print('Read screen names: %s' % screen_names)
    twitter = get_twitter()
    users = get_users(twitter,screen_names)
    num_tweets = gather_tweets(twitter,users)
    read_tweets('vaibhavtweets.txt')
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    
    print('AFINN:', afinn_sentiment(read_tweets('vaibhavtweets.txt'), afinn, verbose=True))

if __name__ == '__main__':
    main()