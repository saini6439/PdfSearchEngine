import datafatch
import json
from base64 import decode

from flask import Flask, render_template, request
from math import log, sqrt
from collections import defaultdict
import nltk
from pdfminer.utils import unicode
from pip._vendor.distlib.compat import raw_input
import glob, os
import io
import re
import chardet
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
PEOPLE_FOLDER = os.path.join('static', 'image')


pdfnos_of_documents = 1;

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----------------------------PDf File Page Search------------------>>>>>>>>>>>>>>>>>>>>>>>

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
@app.route('/')
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Team_logo.png')
    return render_template("index.html",img=full_filename)

@app.route('/pdfsearch',methods = ['POST', 'GET'])
def pdfsearch():
   full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Team_logo.png')
   if request.method == 'POST':
      pdfname = request.form
      searchfiletext = pdfname.to_dict(flat=False)['pdfnamearea'][0]
      searchtext = pdfname.to_dict(flat=False)['query'][0]


   return render_template("pdfsearch.html", result=searchfiletext,img=full_filename,query=searchtext)



@app.route('/result',methods = ['POST', 'GET'])
def result():
   full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Team_logo.png')
   if request.method == 'POST':
      query = request.form
      searchtext = query.to_dict(flat=False)['url'][0]
      #print(decttext)
      query_list = datafatch.get_tokenized_and_normalized_list(searchtext)
      # print("Query List", query_list)
      query_vector = datafatch.create_vector_from_query(query_list)
      # print("Quey Vector", query_vector)
      datafatch.get_tf_idf_from_query_vect(query_vector)
      result = datafatch.get_result_from_query_vect(query_vector)
      countfile = 0;
      for value in result:

          if value[1] > 0:
              countfile = countfile + 1;
      return render_template("searchfile.html",result = reversed(result),img=full_filename,searchquery=searchtext,searchedfiles=countfile)


@app.route('/pdfsearched',methods = ['POST', 'GET'])
def pdfsearched():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Team_logo.png')
    pdfinverted_index = defaultdict(list)
    pdfvects_for_docs = []  # we will need nos of docs number of vectors, each vector is a dictionary
    pdfdocument_freq_vect = {}  # sort of equivalent to initializing the number of unique words to 0
    pdftotaldoc = 0;
    pagetext = []
    if request.method == 'POST':

      pdfname = request.form
      searchfiletext = pdfname.to_dict(flat=False)['pdfnamearea'][0]
      query=pdfname.to_dict(flat=False)['searchquery'][0]

      print(type(searchfiletext))

      def pdfextract_text_by_page(pdf_path):
          with open("corpus/" + pdf_path, 'rb') as fh:
              for page in PDFPage.get_pages(fh,
                                            caching=True,
                                            check_extractable=True):
                  resource_manager = PDFResourceManager()
                  fake_file_handle = io.StringIO()
                  converter = TextConverter(resource_manager, fake_file_handle)
                  page_interpreter = PDFPageInterpreter(resource_manager, converter)
                  page_interpreter.process_page(page)
                  text = fake_file_handle.getvalue()
                  yield text
                  # close open handles
                  converter.close()
                  fake_file_handle.close()

      def pdfextract_text(pdf_path):
          text = ""
          count = 0;
          for page in pdfextract_text_by_page(pdf_path):
              # # print(page)
              count = count + 1
              # print(count)
              # print(page)
              # # print()
              global pdfnos_of_documents
              pdfnos_of_documents = pdfnos_of_documents + 1;
              text += page
              pagetext.append(page)
              # print(pagetext)
              # print("nos_of_documents",nos_of_documents)
              token_list = pdfget_tokenized_and_normalized_list(page)
              # print("token list", token_list)
              vect = pdfcreate_vector(token_list)
              # print("vect", vect)
              pdfvects_for_docs.append(vect)
              # print("vects_for_docs",vects_for_docs)
          # print(count)

          return text

      # creates a vector from a query in the form of a list (l1) , vector is a dictionary, containing words:frequency pairs
      def pdfcreate_vector_from_query(l1):
          vect = {}
          for token in l1:
              if token in vect:
                  vect[token] += 1.0
              else:
                  vect[token] = 1.0
          return vect

      # name is self explanatory, it generates and inverted index in the global variable inverted_index,
      # however, precondition is that vects_for_docs should be completely initialized
      def pdfgenerate_inverted_index():
          count1 = 0
          for vector in pdfvects_for_docs:
              for word1 in vector:
                  pdfinverted_index[word1].append(count1)
              count1 += 1

      # it updates the vects_for_docs global variable (the list of frequency vectors for all the documents)
      # and changes all the frequency vectors to tf-idf unit vectors (tf-idf score instead of frequency of the words)
      def pdfcreate_tf_idf_vector():
          vect_length = 0.0
          for vect in pdfvects_for_docs:
              for word1 in vect:
                  word_freq = vect[word1]
                  temp = pdfcalc_tf_idf(word1, word_freq)
                  vect[word1] = temp
                  vect_length += temp ** 2

              vect_length = sqrt(vect_length)
              for word1 in vect:
                  vect[word1] /= vect_length

      # note: even though you do not need to convert the query vector into a unit vector,
      # I have done so because that would make all the dot products <= 1
      # as the name suggests, this function converts a given query vector
      # into a tf-idf unit vector(word:tf-idf vector given a word:frequency vector
      def pdfget_tf_idf_from_query_vect(query_vector1):
          vect_length = 0.0
          for word1 in query_vector1:
              word_freq = query_vector1[word1]
              if word1 in pdfdocument_freq_vect:  # I have left out any term which has not occurred in any document because
                  query_vector1[word1] = pdfcalc_tf_idf(word1, word_freq)
              else:
                  query_vector1[word1] = log(1 + word_freq) * log(
                      pdfnos_of_documents)  # this additional line will ensure that if the 2 queries,
                  # the first having all words in some documents,
                  #   and the second having and extra word that is not in any document,
                  # will not end up having the same dot product value for all documents
              vect_length += query_vector1[word1] ** 2
          vect_length = sqrt(vect_length)
          if vect_length != 0:
              for word1 in query_vector1:
                  query_vector1[word1] /= vect_length

      # precondition: word is in the document_freq_vect
      # this function calculates the tf-idf score for a given word in a document
      def pdfcalc_tf_idf(word1, word_freq):
          return log(1 + word_freq) * log(pdfnos_of_documents / pdfdocument_freq_vect[word1])

      # define a number of functions,
      # function to to read a given document word by word and
      # 1. Start building the dictionary of the word frequency of the document,
      #       2. Update the number of distinct words
      #  function to :
      #       1. create the dictionary of the term freqency (number of documents which have the terms);

      # this function returns the dot product of vector1 and vector2
      def pdfget_dot_product(vector1, vector2):

          # print("vectore 1:",vector1,"vectore 2:",vector2)
          if len(vector1) > len(vector2):  # this will ensure that len(dict1) < len(dict2)
              temp = vector1
              vector1 = vector2
              vector2 = temp
          keys1 = vector1.keys()
          keys2 = vector2.keys()
          sum = 0
          for i in keys1:
              if i in keys2:
                  sum += vector1[i] * vector2[i]

          return sum

      # this function returns a list of tokenized and stemmed words of any text
      def pdfget_tokenized_and_normalized_list(doc_text):
          # return doc_text.split()
          tokens = nltk.word_tokenize(doc_text)
          ps = nltk.stem.PorterStemmer()
          stemmed = []
          for words in tokens:
              stemmed.append(ps.stem(words))
          return stemmed

      # creates a vector from a list (l1) , vector is a dictionary, containing words:frequency pairs
      # this function should not be called to parse the query given by the user
      # because this function also updates the document frequency dictionary
      def pdfcreate_vector(l1):
          vect = {}  # this is a dictionary
          global document_freq_vect

          for token in l1:
              if token in vect:
                  vect[token] += 1
              else:
                  vect[token] = 1
                  if token in pdfdocument_freq_vect:
                      pdfdocument_freq_vect[token] += 1
                  else:
                      pdfdocument_freq_vect[token] = 1
          return vect



      # this function takes the dot product of the query with all the documents
      #  and returns a sorted list of tuples of docId, cosine score pairs
      def pdfget_result_from_query_vect(query_vector1):
          # print("Enter into query vector")
          parsed_list = []
          # print("nos_of_documents",nos_of_documents)
          print("pdfvaects for docs",pdfvects_for_docs)
          for i in range(pdfnos_of_documents):
              if i < pdfnos_of_documents - 3:
                  print("pdfvects_for_docs",pdfvects_for_docs[i + 1])
                  print("value of i:",i,"pdfnos_of_documents",pdfnos_of_documents)
                  dot_prod = pdfget_dot_product(query_vector1, pdfvects_for_docs[i + 1])

                  # print("Doc vector product",dot_prod)
                  parsed_list.append((pagetext[i + 1], dot_prod))
                  # print("lambda function call and parsed list",parsed_list)
                  parsed_list = sorted(parsed_list, key=lambda x: x[1])

                  # print("after lambda function and parsed list",parsed_list)

          return parsed_list

      pdfextract_text(searchfiletext[:-26])
      pdfgenerate_inverted_index()

      # changes the frequency values in vects_for_docs to tf-idf values
      pdfcreate_tf_idf_vector()

      # print("Here is a list of 15 tokens along with its docIds (sorted) in the inverted index")
      count = 1
      for word in pdfinverted_index:
          if count >= 16:
              break
          # print('token num ' + str(count) + ': ' + word + ': '),
          for docId in pdfinverted_index[word]:
              print(str(docId) + ', '),
          print()
          count += 1



      query_list = pdfget_tokenized_and_normalized_list(query)
      print("Query List pdf search", query_list)
      query_vector = pdfcreate_vector_from_query(query_list)

      print("Quey Vector pdf search", query_vector)
      # print("in into tf idf")
      pdfget_tf_idf_from_query_vect(query_vector)
      # print("outfrom get tf idf")
      result_set = pdfget_result_from_query_vect(query_vector)
      global pdfnos_of_documents;
      pdfnos_of_documents=1
      print(pdfnos_of_documents)
      print(result_set)
      count=0;
      for value in result_set:
          if value[1]>0:
              count=count+1;
      return render_template("pdfsearch.html",result=searchfiletext, output=reversed(result_set),img=full_filename,searchquery=query,searchresults=count)

if __name__ == '__main__':

    app.run()
