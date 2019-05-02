# cs5293sp19-project2

Unredactor.py:

In unredactor program, I trained the supervised machine learning model with the training files( taking input as ‘features of name’ and output as ‘name’) and tested the model with redacted test files (taking ‘features of redacted names of redacted files’ for prediction as x_test) and gives an output file containing predicted names . The undredactor command expects atleast on input to train the model with.
The following are the activities performed in unredactor program:

Step 1.dotrainingextraction(glob_text)

I have taken input as training files path(C:/Users/Shiva SRK/Documents/train/*.txt) and returns list of traininglistoffilesdata. For this project I have used 10,000 files for training which are stored in train/ folder.
I used  the glob.glob() to read multiple files from folder.

Step 2. find_entity(totaltrainingdata):

Input- Passed list of all training documents which are extracted from dotrainingextraction() method.
Output- returns dictionary containing features of names in all the training documents.
Features- number of words in a name, length of first word in a name, length  of second word in a name, length  of third word in a name, length of the name
Step 3. Training the model :

Training the Supervised machine learning model with extracted features of name as input (x_train) and names as output. 
Training data contains list of dictionaries, I am taking x_train with ‘features as names’ and y_train as ‘names’
y_train = []
x_train = []
for item in trainingdata:
    # print(name_dict['name'])
    y_train.append(item['name'])
    del item['name']
    x_train.append(item)

I used the DictVectorizer to transform list of dicts in x_train to compatible array 
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train)
I used the np.array() method to convert list of names into compatible array of names.

y_train = np.array(y_train)

I have used naïve_baye’s classifier – MultinomialNB() , which is trained with x_train(input) and y_train(output) through the fit() method .


Step 4.  dotestingextraction(glob_text)

I have taken input as testing files path(C:/Users/Shiva SRK/Documents/train/*.txt) and returned list of testinglistoffilesdata. For this project I have used 5,000 files for testing which are stored in test/ folder.
I used  the glob.glob() method to read multiple files from this folder.

Step 5. get_redactednameentities():

testinglistoffilesdata (output from dotestingextraction() method) is taken as input  and performed names redaction on  all the testing files and returned list  of redacted documents.

Step 6. ExtractFeatures_redact_data():

Input- Passed list of all redacted documents which are extracted from get_redactednameentities() method.
Output- returns dictionary containing features of names in all the redacted documents.
Features- number of words in a name, length of first word in a name, length of second word in a name, length of third word in a name, length of the name. 
In this, I found out the redacted name by using █ blocks. I have split the redacted data using space as the separator and checked whether the item first element starts with █  block or not.


Step 7. Testing the model with redacted names features

I this, I used the DictVectorizer to transform list of dicts in x_test to compatible array and passed to the model to predict redacted names.
x_test = vec.fit_transform(x_test)
y_pred =  model.predict(x_test)

I have used only one nearest neighbour to the name while predicting.

Bugs: -Errors will occur if the training and test files don’t contain at least one name.






