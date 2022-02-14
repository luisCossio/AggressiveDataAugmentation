import argparse
import numpy as np
from google_images_download import google_images_download


class downloader:
    def __init__(self, Max_downloads=100):
        self.max_download = Max_downloads
        self.size = 'large'

    def download_queries(self, queries):
        response = google_images_download.googleimagesdownload()

        for query in queries:
            self.downloadimages(query, response)

    def downloadimages(self, query, response):
        # keywords is the search query
        # format is the image file format
        # limit is the number of images to be downloaded
        # print urs is to print the image file url
        # size is the image size which can
        # be specified manually ("large, medium, icon")
        # aspect ratio denotes the height width ratio
        # of images to download. ("tall, square, wide, panoramic")
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit": self.max_download,
                     "print_urls": False,
                     "size": self.size
                     # "aspect_ratio": "panoramic",
                     }
        try:
            response.download(arguments)

        # Handling File NotFound Error
        except FileNotFoundError:
            print("Invalid big size for query {:s}".format(query))
            arguments = {"keywords": query,
                         "format": "jpg",
                         "limit": self.max_download,
                         "print_urls": False,
                         "size": "large"}

            # arguments = {"keywords": query}

            # Providing arguments for the searched query
            try:
                # Downloading the photos based
                # on the given arguments
                response.download(arguments)
            except:
                pass


def randomize_words(people, context, additive=[''], probability_additive=0.5, n_phrases=20):
    if len(additive) == 0:
        probability_additive = 0
    elif len(additive[0]) == 0:
        probability_additive = 0
    words = []
    people_words = np.random.choice(people, size=[n_phrases])
    context_words = np.random.choice(context, size=[n_phrases])
    extras = np.random.choice(additive, size=[n_phrases])
    random_prob = np.random.uniform(0, 1, size=[n_phrases])
    results = random_prob < probability_additive
    for i in range(n_phrases):
        words += [people_words[i] + ' ' + context_words[i]]
        if results[i]:
            words[-1] += ' ' + extras[i]
    return words


def main(queries, N_images):
    dw = downloader(Max_downloads=N_images)
    dw.download_queries(queries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='downloader_images.py')
    # parser.add_argument('--queries', nargs='+', help='<Required> List of queries to load')
    parser.add_argument('--max-images', type=int, help='number images to look for per query', default=100)
    opt = parser.parse_args()
    # opt.queries = ['posing profile', 'futbol partido', 'party fete', 'restaurant outdoors', 'news amazing',
    #                'sport jumping', 'dancing class', 'scout groups', 'gathering meeting', 'dog owner',
    #                'manifestation streets', 'class exposition', 'work business', 'salesman great',
    #                'fashionable outfits',
    #                'community meeting', 'meeting alcoholics', 'retirement house',
    #                'golf shots', 'sport shots',
    #                'sailors crew', 'nurses hospital', 'people streets', 'marching band',
    #                'demonstration event',
    #                'people accident', 'funeral', 'cheering', 'group talking', 'interview people',
    #                'driver car',
    #                'stock trader', 'award receiving', 'day ceremony', 'concert playing', 'couple',
    #                'family trip', 'festival fans', 'picnic park', 'shoppers local', 'soldiers patrol',
    #                'soldiers drilling',
    #                'spa therapist', 'student school', 'surgeon hospital',
    #                'waiter waitress', 'construction labor',
    #                'basketball interschool']

    np.random.seed(1236933)

    subject = ['people', 'meeting', 'gathering', 'she is', 'local', 'award', 'couple',
               'elderly', 'children', 'neighbourly', 'shoppers', 'waitress', 'nurse', 'doctor',
               'model', 'employee', 'engineer', 'trader', 'daughter', 'social', 'professor', 'journalist',
               'politician', 'vendor', 'tourists', 'student', 'immigrants', 'sick', 'inhabitant',
               'mother', 'player', 'contestant', 'father', 'adult', 'teenager', 'retiree', 'citizen',
               'consumer', 'celebrity', 'author', 'veterinary', 'reporter', 'detective', 'crowd', 'family', 'group',
               'soldier', 'patrol', 'fan', 'runners', 'instructor', 'police', 'coach', 'students', 'jury', 'child',
               'experts', 'squad', 'clown', 'driver', 'actor', 'actress', 'miller', 'carpenter', 'public',
               'grandchildren', 'youth', 'influencer', 'operator', 'supervisor',''
               'minors', 'workers', 'fisherman']

    combinations = ['golf', 'scout', 'soccer', 'futbol', 'football', 'cheering', 'caring',
                    'promising', 'waiting', 'sewing', 'exposition', 'office', 'lines',
                    'garden', 'basketball', 'podcast', 'sitting', 'school', 'business', 'sharing', 'jumping',
                    'house', 'retirement', 'authorities', 'field', 'parade', 'fabric', 'factory',
                    'company', 'skating', 'manufacture', 'teaching', 'streets', 'neighbourhood', 'hospital',
                    'passionate', 'working', 'having', 'learning', 'bachelor', 'pulling',
                    'cutting', 'threatening', 'arrest', 'produce', 'bringing', 'violent',
                    'chatting', 'handling', 'hold', 'walking', 'standing', 'punching', 'driving',
                    'mounting', 'zoo', 'rural', 'shop', 'tired', 'angry', 'workout', 'lifting',
                    'campus', 'prison', 'donate', 'control', 'aero', 'capitan', 'reader',
                    'audience', 'holder', 'wielding', 'holding', 'poor', 'paying', 'costume',
                    'comic', 'classroom', 'TV', 'Series', 'casual', 'wanting', 'struggling', 'room',
                    'ground', 'base', 'metro', 'museum', 'camping', 'supervising', 'balancing', 'beach',
                    'picture group', 'marching', 'parading', 'grouping', 'regrouping', 'concert', 'musical',
                    'trip', 'festival', 'drilling', 'seat', 'table', 'restaurant', 'outdoors', 'race', 'aerobics',
                    'raid', 'running', 'choreography', 'graduation', 'bar', 'club', 'houseparty', 'contest',
                    'scandal', 'prom', 'barbecue', 'food', 'fast', 'trekking', 'class', 'party', 'birthday', 'kinder',
                    'reunion', 'think-tank', 'newspaper', 'fishing', 'pedestrian']

    extras = ['trending', 'news', 'greatest', 'authorities', 'current', 'average', 'high', 'great',
              'low', 'latest', 'inspiring', 'sadly', 'biography', 'show', 'unique', 'most',
              'interview', 'terrible', 'never', 'got', 'famous', 'top', 'painful', 'next',
              'small', 'poor', 'successful', 'impoverished', 'declared', 'pool', 'better', 'tattoo',
              'effect', 'implicit', 'way', 'found', 'irresponsible', 'blurry', 'still', 'forum',
              'city', 'sneaky', 'scientific', 'animal', 'outside', 'consequences', 'open', 'safe',
              'eighties', 'back', '90s', 'disappointing', 'future', 'understandably', 'attention',
              'facing', 'chill', '2000', 'novel', 'capitalism', 'rich', 'life', 'shocking',
              'potential', 'unbelievable', 'experience', 'first', 'decade', 'class', 'classic', 'kill',
              'wheel', 'swing', 'bizarre', 'dog', 'cat', 'pet', 'point', 'take', 'group', 'war', 'past',
              'shadow', 'area', 'accomplished',
              'control', 'friday', 'wanted', 'struggling', 'crowded', 'background', 'away',
              'article', 'opening', 'formation', 'organized', 'sport', 'marathon', 'against', 'urban', 'support',
              'difference', 'grand', 'release', 'roaring', 'new', 'letters', 'make up', 'fashion', 'scene', 'station']

    queries = randomize_words(subject, combinations, extras)
    # queries = ['arbol manzanas','fruit apple tree','orchard tree apple','plantation apples','pomme arbre',
    #            'apples red tree','gala apple tree','green apple tree','apple dataset']

    main(queries, opt.max_images)
