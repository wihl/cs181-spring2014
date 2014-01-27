import csv

def load_books(filename='books.csv'):
    """Load in the book data.

    Keyword arguments:
    filename -- the name of the csv file (default 'books.csv')

    Returns a list of dictionaries, e.g.,
    [   {   'author': 'Richard Bruce Wright',
            'isbn': '0002005018',
            'publisher': 'HarperFlamingo Canada',
            'title': 'Clara Callan',
            'year': 2001},
        {   'author': "Carlo D'Este",
            'isbn': '0060973129',
            'publisher': 'HarperPerennial',
            'title': 'Decision in Normandy',
            'year': 1991},
        {   'author': 'Amy Tan',
            'isbn': '0399135782',
            'publisher': 'Putnam Pub Group',
            'title': "The Kitchen God's Wife",
            'year': 1991},
        ... ]
    """

    with open(filename, 'rb') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        books = []
        for row in reader:
            books.append( { 'isbn':      row[0],
                            'title':     row[1],
                            'author':    row[2],
                            'publisher': row[3],
                            'year':      int(row[4]) })
    return books

def load_users(filename='users.csv'):
    """Load in the user data.

    Keyword arguments:
    filename -- the name of the csv file (default 'users.csv')

    Returns a list of dictionaries, e.g.,
    [   {   'age': 0, 'location': 'Timmins, Ontario, Canada', 'user': 3527},
        {   'age': 42, 'location': 'Franktown, Colorado, USA', 'user': 6948},
        {   'age': 57, 'location': 'Ligonier, Pennsylvania, USA', 'user': 11942},
        {   'age': 27, 'location': 'Porto, Porto, Portugal', 'user': 7660},
        ... ]
    """

    with open(filename, 'rb') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        users = []
        for row in reader:
            users.append( { 'user':     int(row[0]),
                            'location': row[1],
                            'age':      int(row[2]) })
    return users

def load_train(filename='ratings-train.csv'):
    """Load in the training data.

    Keyword arguments:
    filename -- the name of the csv file (default 'ratings-train.csv')

    Returns a list of dictionaries, e.g.,
    [ {'rating': 3, 'isbn': '0449911004', 'id': 247128, 'user': 2178},
      {'rating': 4, 'isbn': '0618129022', 'id': 197566, 'user': 943},
      {'rating': 5, 'isbn': '0930289595', 'id': 287153, 'user': 1417},
      {'rating': 4, 'isbn': '0312960808', 'id': 255840, 'user': 6665},
      ... ]
    """

    with open(filename, 'rb') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        ratings = []
        for row in reader:
            ratings.append( { 'id':     int(row[0]),
                              'user':   int(row[1]),
                              'isbn':   row[2],
                              'rating': int(row[3]) })
    return ratings

def load_test(filename='ratings-test.csv'):
    """Load in the test queries.

    Keyword arguments:
    filename -- the name of the csv file (default 'ratings-test.csv')

    Returns a list of dictionaries, e.g.,
    [   {   'id': 268752, 'isbn': '0446610038', 'user': 3389},
        {   'id': 80629, 'isbn': '0345306880', 'user': 304},
        {   'id': 189135, 'isbn': '0440224764', 'user': 546},
        {   'id': 270511, 'isbn': '0451524551', 'user': 5153},
        {   'id': 179535, 'isbn': '0425170349', 'user': 599},
        ... ]
    """
    
    with open(filename, 'rb') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        queries = []
        for row in reader:
            queries.append( { 'id':     int(row[0]),
                              'user':   int(row[1]),
                              'isbn':   row[2] })
    return queries

def write_predictions(preds, filename):
    """Write out a prediction file.

    Arguments:
    preds -- a list of dictionaries corresponding to test queries,
             but with a 'rating' entry also for the prediction:
    [ {'rating': 4.070495, 'isbn': '1843606127', 'id': 232545, 'user': 1948},
      {'rating': 4.070495, 'isbn': '0375503862', 'id': 90221, 'user': 3794},
      {'rating': 4.070495, 'isbn': '0451167716', 'id': 54492, 'user': 6467},
      {'rating': 4.070495, 'isbn': '0345339703', 'id': 100429, 'user': 4342},
      {'rating': 4.070495, 'isbn': '0786885688', 'id': 56837, 'user': 10886},
      ... ]

    filename -- the file to which the predictions should be written
    """

    with open(filename, 'wb') as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id','Prediction'])
        for pred in preds:
            writer.writerow([pred['id'], pred['rating']])

