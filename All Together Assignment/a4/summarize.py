from collect import read_screen_names, get_twitter, get_users
from cluster import girvan_partitioning
from classify import gather_tweets, afinn_sentiment, read_tweets

def main():
    users = []
    num_tweets = 0
    communities = []

    f = open('summary.txt', 'w')

    twitter = get_twitter()
    screen_names = read_screen_names('teams.txt') 
    users = get_users(twitter,screen_names)
    num_tweets = gather_tweets(twitter, screen_names) 
    communities = girvan_partitioning(g, 10, 15)
    read_tweets('vaibhavtweets.txt')
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    f.write("Number of users collected:", len(screen_names))
    f.write("Number of messages collected:", num_tweets)
    f.write("Number of communities discovered:", len(communities))
    f.write("Community 1: ", len(communities[0]))
    f.write("Community 2: ", len(communities[1]))
    f.write("Number of instances per class found:", afinn_sentiment(read_tweets('vaibhavtweets.txt'), afinn, verbose=True))

if __name__ == '__main__':
    main()
    
