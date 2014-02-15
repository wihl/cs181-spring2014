import re
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# MovieData is an object to encapsulate the metadata and reviews associated with 
# a single movie. It is created from an "instance" element contained in either 
# train.xml or testcases.xml, and should allow you to ignore parsing the xml and 
# focus on feature extraction and learning.
class MovieData(object):
    """
    A MovieData object may have any of the following attributes (some may be missing
    from any given object):
      AC - Austin Chronicle review text (string)
      BO - Boston Globe review text (string)
      CL - LA Times review text (string)
      EW - Entertainment Weekly review text (string)
      NY - New York Times review text (string)
      VA - Variety review text (string)
      VV - Village Voice review text (string)
      actors - list of actor strings
      authors - list of author strings
      christmas_release - boolean
      company - production company string
      directors - list of director strings
      genres - list of genre strings
      highest_grossing_actor - list of actor strings
      highgest_grossing_actors_present - boolean
      id - unique string id of movie
      independence_release - boolean
      labor_release - boolean
      memorial_release - boolean
      name - official movie name string
      num_highest_grossing_actors - float
      num_oscar_winning_actors - float
      num_oscar_winning_directors - float
      number_of_screens - float
      origins - list of country-name strings
      oscar_winning_actor - list of actor strings
      oscar_winning_actors_present - boolean
      oscar_winning_director - list of director strings
      oscar_winning_directors_present - boolean
      production_budget - float
      rating - string
      release_date - string
      running_time - float
      summer_release - boolean
      target - float
    """
    numeric_fields = ["running_time", "production_budget"] # numeric fields that don't start with "num"
    implicit_list_atts = ["oscar_winning_director", "oscar_winning_actor", "highest_grossing_actor"]
    reviewers = ["AC","BO","CL","EW","NY","VA","VV"]
    
    def __init__(self, inst_el):
        """
        inst_el is an ElementTree element representing a movie instance, extracted from
        train.xml or testcases.xml
        """
        self.id = inst_el.attrib['id']
        for child_el in inst_el:
            try:
                if child_el.tag == "regy": # opening week revenue
                    self.target = float(child_el.attrib['yvalue'])
                elif child_el.tag == "text": # reviews
                    self.__dict__[child_el.attrib['tlabel']] = asciify(child_el.text)
                elif child_el.tag.endswith('release'): # special weekend releases
                    self.__dict__[child_el.tag] = False if child_el.text.strip() == "false" else True
                elif child_el.tag in self.implicit_list_atts: # these can appear multiple times w/ different vals
                    if hasattr(self, child_el.tag):
                        self.__dict__[child_el.tag].append(asciify(child_el.text))
                    else:
                        self.__dict__[child_el.tag] = [asciify(child_el.text)]
                elif len(child_el) > 0: # list (e.g., actors, genres)
                        self.__dict__[child_el.tag] = [asciify(el.text) if el.text is not None else "" for el in child_el]
                elif len(child_el.attrib) == 0 and child_el.text is None: # just a predicate
                    self.__dict__[child_el.tag] = True
                elif len(child_el.attrib) == 0 and (child_el.tag.startswith('num') or child_el.tag in self.numeric_fields):
                    self.__dict__[child_el.tag] = float(child_el.text.replace(",","").replace("$",""))
                elif len(child_el.attrib) == 0:
                    self.__dict__[child_el.tag] = asciify(child_el.text)
            except Exception:
                print ET.tostring(child_el)
                import sys
                sys.exit(1)

# a function for removing non-ascii characters.
# from http://stackoverflow.com/questions/1342000/how-to-replace-non-ascii-characters-in-string
def asciify(s):
    return "".join(i for i in s if ord(i)<128)

# returns True if s is a string representing a valid floating pt number
def non_numeric(s):
    try:
        n = float(s)
        return False
    except ValueError:
        return True

# if you would like to use stopwords in your feature engineering, consider using 
# the stopwords in english.stop (from http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop)
# as follows:
#with open("english.stop") as f:
#    stop_words = set([line.strip() for line in f.readlines()])

# a regular expression for identifying punctuation in reviews
punct_patt = re.compile('[\'\.,:\?;!"\(\)\[\]\$@%]')

# a function for writing predictions in the required format
def write_predictions(predictions, ids, outfile):
    """
    assumes len(predictions) == len(ids), and that predictions[i] is the
    predicted opening revenue for movie corresponding to ids[i]
    outfile will be overwritten
    """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, movie_id in enumerate(ids):
            f.write("%s,%f\n" % (movie_id, predictions[i]))
