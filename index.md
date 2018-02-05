### *Project Proposal:* **Dr.Job -- A Smart Job Recommendation Engine**

Fresh graduates face many problems while applying for jobs and are having millions of doubts in their mind. It not hard to believe that fresh graduates, even Ph.D. graduate don’t know what career paths lie in front of them at the moment of job hunting. A Ph.D. who got training in engineering might be a good match for a business analyst position with the analytical skills he/she gained during more than 5 years in graduate school. But the real story is many of them didn’t even think of looking for jobs in fields outside of  their tiny research area. 

Ironically, the difficulties of finding the perfect jobs lie not in the scarcity of job information, but overwhelming of online information for job hunting thanks to the blooming of information technology. Numerous websites exist for displaying job postings, among them are Monster.com, Indeed.com, and CareerBuilder.com, in addition to LinkedIn.com job section. One search of “Data Scientists” on Indeed.com gives you 20,000+ results, let alone other websites. It’s just impossible to click and check all the positions by hand. Although all of these websites provide search results filter feature to help ease the job hunting process, it didn’t solve the fundamental problem of identifying the right career,and therefore choose the correct key words. Some of these websites even provide preliminary job recommendation features such as that offered by CareerBuilder.com. However, a recommender system like that makes job or company recommendations based primarily on users’ browsing history, rather than on the true background of the users. 


The idea proposed here is to design a smart web and app platform for job seeker to find the right jobs with the help of machine learning and historical employment data.  There are subtle difference between a “dream job” and a “best-match job”. The purposes of this proposed job recommendation engine include:
to give user a clear image of where they stand among other applicants by comparing their profiles
to provide users a summary of job statistics, such as where has the most job opening, and what skill sets do they require
to introduce a “Job Match Rating System” that utilize latent semantics based information retrieval techniques to rate job postings so that users can focus on jobs that suit them best.
to provide a profile refining recommendation based on the difference between user profile of requirements of their dream jobs. 
The part of the summary of job statistics is implemented as a demo for this Data Incubator challenge problem. The rest will be completed at 2016 Data Incubator Cohort. 


### Describe Data Source
Historical employment data is scraped from job posting and professional social network websites by specially designed "web spiders". Also job postings are collected from various job posting websites. [Indeed.com](www.indeed.com) and [LinkedIn](www.linkedin.com) also provide APIs for job searching. These features may be employed in the future.
Python Spider code for collecting data used in this project so far is attached:
~~~
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------------------------
import urllib2
from bs4 import BeautifulSoup
import pandas as pd
from nltk.corpus import stopwords
import re
import pickle
from time import sleep # To prevent overwhelming the server between connections
from collections import Counter # Keep track of our term counts


# create a dataframe of job listings from lists of the titles, companies, locations, and links
def createJobListingsDF(titles,companies,cities,states,links):#,descriptions):
    jobDict = {
    'Title':titles,
    'Company':companies,
    'City':cities,
    'State':states,
    'Link':links
    #'Description':descriptions
    }
    
    jobListings = pd.DataFrame(jobDict)
    return jobListings

    
# Webscraping Indeed.com    
#-------------------------------------------------------------------------------------------------
# generate a list with all page urls
def countIndeedJobs(nPostsToShow):    
    # combine base url with user-defined search terms
    # first page    
    baseUrl = 'http://www.indeed.com/jobs?q=' + search_url +'&filter=0&start='
    pagesUrl = urllib2.urlopen(baseUrl)
    soup = BeautifulSoup(pagesUrl)
    # get the total number of all pages 
    numberInString = soup.find('div', {'id':'searchCount'}).text[16:]
    
    totalListings = int(numberInString.replace(',',''))
    #nPostsToShow = 30
    print min(nPostsToShow,totalListings)
    pages = range(0, min(nPostsToShow,totalListings), 10)
    myUrls = []
    # generate urls for page 2 and above
    for apage in pages:
        myUrls.append(baseUrl + str(apage))
    return myUrls

# Parse webpage and return lists containing titles, companies, cities, states, and links
def getIndeedPage(aUrl):
    jobsPage = urllib2.urlopen(aUrl)
    soup = BeautifulSoup(jobsPage)
    jobs = soup.findAll('td',{'id':'resultsCol'})
    titles = []
    companies = []
    cities = []
    states = []
    links = []
    #descriptions =[]
    for job in jobs:
        titleBlocks = job.findAll('div',{'itemtype':'http://schema.org/JobPosting'})
        for titleBlock in titleBlocks:
            title = titleBlock.find('a')['title']
            titles.append(title)
            link = 'www.indeed.com' + titleBlock.find('a')['href']
            links.append(link)
        companyBlocks = job.findAll('span',{'itemtype':'http://schema.org/Organization'})            
        for companyBlock in companyBlocks:            
            company = companyBlock.get_text('span',{'itemprop':'name'})
            companies.append(company)
        locationBlocks = job.findAll('span',{'itemtype':'http://schema.org/Postaladdress'})            
        for locationBlock in locationBlocks:            
            location = locationBlock.get_text('span',{'itemprop':'addressLocality'})
            city, space, state = location.partition(', ')            
            cities.append(city)
            states.append(state[:2])       
    return [titles,companies,cities,states,links]

# Iterate through all webpages and convert lists to dataframe
def getIndeedJobs(nPostsToShow,titles = [],companies = [],cities = [], states = [],links = []):
    myUrls = countIndeedJobs(nPostsToShow)
    statusCount=0 
    for aUrl in myUrls:
        data = getIndeedPage(aUrl)
        #print data[0]
        titles = titles + data[0]
        companies = companies + data[1]
        cities = cities + data[2]
        states = states + data[3]
        links = links + data[4]
        statusCount += 1
        print statusCount
    print 'start grabbing indeed.com jobs...'
    jobDict = {
    'Title':titles,
    'Company':companies,
    'City':cities,
    'State':states,
    'Link':links
    #'Description':descriptions
    }
    return jobDict
    #allJobs = createJobListingsDF(titles,companies,cities,states,links)
    print 'finished grabbing indeed.com Jobs!'
    #return allJobs
    
def getIndeedJobsFromUrls(urls):
    titles = []
    companies = []
    cities = []
    states = []
    links = []
    myUrls = urls
    statusCount=0 
    for aUrl in myUrls:
        data = getIndeedPage(aUrl)
        #print data[0]
        titles = titles + data[0]
        companies = companies + data[1]
        cities = cities + data[2]
        states = states + data[3]
        links = links + data[4]
        statusCount += 1
        print statusCount
    print 'start grabbing indeed.com jobs...'
    jobDict = {
    'Title':titles,
    'Company':companies,
    'City':cities,
    'State':states,
    'Link':links
    #'Description':descriptions
    }
    print 'finished grabbing indeed.com Jobs!'
    return jobDict
    #allJobs = createJobListingsDF(titles,companies,cities,states,links)
    
    #return allJobs

def getDescriptions(links):
    descriptions =[]
    for link in links:
        description = getOneDescription(link)
        descriptions.append(description)
        print "processed one more job link!"
    return descriptions
    
def getOneDescription(link):
    jobUrl = 'http://'+link
    try:
        page = urllib2.urlopen(jobUrl).read() # Connect to the job posting
    except: 
        return
        #page = urllib2.urlopen(jobUrl)
    _soupJob = BeautifulSoup(page)
    for script in _soupJob(["script", "style"]):
        script.extract() # Remove these two elements from the BS4 object
    text=_soupJob.get_text()
    #print text
    #lines = (line.strip() for line in text.splitlines()) # break into lines
    
    lines = (line.strip() for line in text.splitlines()) # break into lines
    chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) # break multi-headlines into a line each
    def chunk_space(chunk):
        chunk_out = chunk + ' ' # Need to fix spacing issue
        return chunk_out  
    text = ''.join(chunk_space(chunk) for chunk in chunks if chunk).encode('utf-8') # Get rid of all blank lines and ends of line
    # Now clean out all of the unicode junk (this line works great!!!)
    try:
        text = text.decode('unicode_escape').encode('ascii', 'ignore') # Need this as some websites aren't formatted
    except:                                                            # in a way that this works, can occasionally throw
        return                                                         # an exception   
    text = re.sub("[^a-zA-Z.+3]"," ", text)  # Now get rid of any terms that aren't words (include 3 for d3.js)
                                                # Also include + for C++
    text = text.lower().split()  # Go to lower case and split them apart
    #print text
    stop_words = set(stopwords.words("english")) # Filter out any stop words
    text = [w for w in text if not w in stop_words]
    return text


#if __name__ == "__main__":   
key_words = 'Data Scientist'          
search_url = '+'.join(key_words.split())            # Set search terms for Indeed and CareerBuilder    
nPostsToShow = 20000 # if inf : to show all job posts
allUrls = countIndeedJobs(nPostsToShow)
jobArray=[]
for i in range(20):
    #urlArray.append(url[i*100:100*(i+1)]
    url1000 =allUrls[i*100:100*(i+1)]
    job1000 = getIndeedJobsFromUrls(url1000)
    jobArray.append(job1000)
    print "finished"+str(1000*(i+1))
    
descriptionArray=[]
titles=[]
companies=[]
cities=[]
states=[]
links=[]

for i in range(19):
    titles = titles + jobArray[i]['Title']
    companies = companies + jobArray[i]['Company']
    cities = cities + jobArray[i]['City']
    states = states + jobArray[i]['State']
    links = links + jobArray[i]['Link']
jobDict = {"Title":titles,"Company":companies,"City":cities,"State":states,"Link":links}
with open('jobDict.csv','wb') as f:
    pickle.dump(jobDict,f)
    
    
filename = "job"+str(i+1)+"000.csv"

descriptionArray=[]
descriptions=[]
allJobLinks = jobDict['Link']
for i in range(190):
    url100 = allJobLinks[100*i:100*(i+1)]
    description100 = getDescriptions(url100)
    print "processed:"+str(i)+" x100"
    descriptionArray.append(description100)
    descriptions=descriptions+description100
with open('descriptions.csv','wb') as f:
    pickle.dump(descriptions,f) 
    
for i in range(20):
    #urlArray.append(url[i*100:100*(i+1)]
    jobLink1000 = jobArray[i]['Link']
    description1000 = getDescriptions(jobLink1000)
    descriptionArray.append(description1000)
    print "finished"+str(1000*(i+1))


job_descriptions=descriptions    
doc_frequency = Counter() # This will create a full counter of our terms. 
[doc_frequency.update(item) for item in job_descriptions] # List comp
    # Now we can just look at our final dict list inside doc_frequency
    # Obtain our key terms and store them in a dict. These are the key data science skills we are looking for
prog_lang_dict = Counter({'R':doc_frequency['r'], 'Python':doc_frequency['python'],
                'Java':doc_frequency['java'], 'C++':doc_frequency['c++'],
                'Ruby':doc_frequency['ruby'],
                'Perl':doc_frequency['perl'], 'Matlab':doc_frequency['matlab'],
                'JavaScript':doc_frequency['javascript'], 'Scala': doc_frequency['scala']})

analysis_tool_dict = Counter({'Excel':doc_frequency['excel'],  'Tableau':doc_frequency['tableau'],
                    'D3.js':doc_frequency['d3.js'], 'SAS':doc_frequency['sas'],
                    'SPSS':doc_frequency['spss'], 'D3':doc_frequency['d3']})  

hadoop_dict = Counter({'Hadoop':doc_frequency['hadoop'], 'MapReduce':doc_frequency['mapreduce'],
            'Spark':doc_frequency['spark'], 'Pig':doc_frequency['pig'],
            'Hive':doc_frequency['hive'], 'Shark':doc_frequency['shark'],
            'Oozie':doc_frequency['oozie'], 'ZooKeeper':doc_frequency['zookeeper'],
            'Flume':doc_frequency['flume'], 'Mahout':doc_frequency['mahout']})

database_dict = Counter({'SQL':doc_frequency['sql'], 'NoSQL':doc_frequency['nosql'],
                'HBase':doc_frequency['hbase'], 'Cassandra':doc_frequency['cassandra'],
                'MongoDB':doc_frequency['mongodb']})
            
overall_total_skills = prog_lang_dict + analysis_tool_dict + hadoop_dict + database_dict # Combine our Counter objects
final_frame = pd.DataFrame(overall_total_skills.items(), columns = ['Term', 'NumPostings']) # Convert these terms to a                                                                                                 # dataframe
    # Change the values to reflect a percentage of the postings 
final_frame.NumPostings = (final_frame.NumPostings)*100/len(job_descriptions) # Gives percentage of job postings                                                                                    #  having that term 
    # Sort the data for plotting purposes
final_frame.sort(columns = 'NumPostings', ascending = False, inplace = True)
    # Get it ready for a bar plot
final_plot = final_frame.plot(x = 'Term', kind = 'bar', legend = None, 
                            title = 'Percentage of Data Scientist Job Ads with a Key Skill ' )#+ city_title)
final_plot.set_ylabel('Percentage Appearing in Job Postings')
fig = final_plot.get_figure() # Have to convert the pandas plot object to a matplotlib object
~~~

### Figure for Collected Data
![Key Skills for Data Scientist Position](https://drive.google.com/open?id=0B9FWcQkjA-1hU1pNMEk2YTlOLUNySTZtY1RDaUI2eHQ5aHZn)
![Key skills for Data Scientist Position](https://drive.google.com/open?id=0B9FWcQkjA-1hU1pNMEk2YTlOLUNySTZtY1RDaUI2eHQ5aHZn)
(https://drive.google.com/file/d/0B9FWcQkjA-1hU1pNMEk2YTlOLUNySTZtY1RDaUI2eHQ5aHZn/view?usp=sharing)
https://drive.google.com/open?id=0B9FWcQkjA-1hU1pNMEk2YTlOLUNySTZtY1RDaUI2eHQ5aHZn