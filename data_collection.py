''' Data Collection '''

import pandas as pd
import numpy as np

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.exporters import CsvItemExporter


from youtube_transcript_api import YouTubeTranscriptApi
from operator import itemgetter
import re


from __future__ import unicode_literals
import speech_recognition as sr
import argparse
import subprocess
import os

import youtube_dl
import librosa
import soundfile as sf
import wave







class archiveorg_spider(scrapy.Spider):
    
    name = 'youtube_link_crawler'
    allowed_domains = ['archive.org']
    start_urls = crawl_target_2016_Q4 

    custom_settings = {      
        'FEED_EXPORT_FIELDS': ["modified_url", "associated_youtube_link"],
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'youtbue_link_output.csv'
    }

    
    
    
    def parse(self, response):
        
        print("processing!!:"+response.url)
        youtube_links = response.xpath('/html//div[@class = "metadata-expandable-list row"]//dd[@class = " breaker-breaker"]/a/text()').extract()
        
        youtube_links = youtube_links[0]
        youtube_links = str(youtube_links).strip();
        
        print("str(youtube_links)", youtube_links);

        scraped_info = {'modified_url': response.url,
                'associated_youtube_link': youtube_links
               }
        
        yield scraped_info




class archive_detail_transcript_crawler(scrapy.Spider):
    
    name = 'archive_detail_transcript_crawler'
    allowed_domains = ['archive.org']
    start_urls = all_links_16Q4 # this can be changed

    custom_settings = {      
        'FEED_EXPORT_FIELDS': ["modified_url", "transcript"],
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'archive_transcript_2016Q4.csv'
    }

    
    def parse(self, response):
        
        try:
            print("processing!!:"+response.url)
            transcripts = response.xpath('/html//main[@id="maincontent"]/div[5]//div[@id="descript"]/text()').extract()

            transcripts = transcripts[0]
            transcripts = str(transcripts).strip();

            print("transcripts", transcripts);

            scraped_info = {'modified_url': response.url,
                    'transcript': transcripts
                   }

        except:
            scraped_info = {'modified_url': response.url,
                    'transcript': "none-webscrappable"
                   }
        
        
        yield scraped_info






class press_release_link_crawler(scrapy.Spider):
    
    name = 'press_release_link_crawler'
    allowed_domains = ['www.presidency.ucsb.edu']
    
    start_urls = hillary_urls # this can be changed

    custom_settings = {      
        'FEED_EXPORT_FIELDS': ["date", "content_link", "content_category"],
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'press_release_metadata.csv'
    }

    
    def parse(self, response):
        
        print("processing!!:"+response.url)
        
        
           #for i in range()
        
        content_date = response.xpath('/html//div[@class = "field-docs-start-date-time"]//span[@class = "date-display-single"]/text()').extract()
        content_date = [link+"분" for link in content_date]  
        
        content_links = response.xpath('/html//div[@class = "views-field views-field-title"]//a/@href').extract()                    
        content_links = ["https://www.presidency.ucsb.edu"+link for link in content_links]
        content_links = [fulllink+"분" for fulllink in content_links]
        
        
        print(content_links)
        
        
        content_category = response.xpath('/html//div[@class = "views-field views-field-title"]//a/text()').extract() 
        content_category = [cat+"분" for cat in content_category]
        

        scraped_info = {'date': content_date, 'content_link': content_links, 'content_category': content_category}

        yield scraped_info




class press_release_text_crawler(scrapy.Spider):
    
    name = 'press_release_text_crawler'
    allowed_domains = ['presidency.ucsb.edu']
    
    start_urls = hillary_links # this can be changed

    custom_settings = {      
        'FEED_EXPORT_FIELDS': ["content_link", "content_type", "content"],
        'FEED_FORMAT': 'json',
        'FEED_URI': 'content_press_release.json'
    }

    def parse(self, response):
        
        print("processing!!:"+response.url)
        
        content = response.xpath('//*[@class="field-docs-content"]/p/text()').extract()
        
        if ( len(content) >= 2 ):    
            
            content_type = "text"
            
            #print("content:", content)
            #print("content_type=",content_type)
            
        elif ( len(content) == 1 ):
           
        
            if  ( content[0] == "\n"): 
                     
                content = response.xpath('/html//div[@class = "field-docs-content"]//center/iframe/@src').extract()
                
                if ( 'youtube' in content[0] ):
                    content_type = "youtube video"                
                else:
                    content_type = "others video"
                        
                
            elif ( content[0] == "To view this video please enable JavaScript, and consider upgrading to a web browser that"  ):
                
                content = response.xpath('/html//div[@class = "field-docs-content"]//video/source/@src').extract()
                content_type = "HTML5 video"
                
                if ( len(content)==0 ):
                    content = response.xpath('/html//div[@class = "field-docs-media-video-hosted"]//iframe/@src').extract()
                    content_type = "HTML5 video"
                
            
            else:
                
                if ( content[0].isspace() == True ):
                
                    content_type = "None"
                
                else:
                    content_type = "text"
                
                
                #print("content:", content)
                #print("len(content)=", len(content))
                #print("content_type=",content_type) 
            
            
           
            
        elif ( len(content) == 0 ) : 

            content = response.xpath('/html//div[@class = "field-docs-content"]//tbody/tr/td/text()').extract()
  
            
            if (len(content)!=0):
                content_type = "tweet"
                #print("content:", content)
                #print("content_type=",content_type)
                
            elif (len(content)==0):

                content = response.xpath('/html//div[@class = "field-docs-content"]//center/iframe/@src').extract()
                
                if (len(content)==1):
                    if ( 'youtube' in content[0] ):
                        content_type = "youtube video"                
                    else:
                        content_type = "others video"                
                
                else:
                    
                    content = response.xpath('/html//div[@class = "field-docs-content"]//p//text()').extract()
                    
                    if (len(content)==0):
                        content_type = "None"
 
                    else:
                        content_type = "text"
                    
                    
                #print("content:", content)
                #print("content_type=",content_type)
            
        scraped_info = {'content_link': response.url, 'content_type': content_type, 'content': content}

        yield scraped_info





# extracting transcripts from youtube videos by unique video ids. 
#  Note: this is for youtube videos which 'have' trasncripts (sub-titles)
def transcript_list_maker(df):
    
    transcript_list = []
    rowsize = len(df)
    
    for i in range(0, rowsize):
        
        link_string = str(df['associated_youtube_link'][i])
        if "www.youtube.com" in link_string:            
            video_id = link_string[-11:]
            
            # if youtube subtitle is disabled, can't extract.
            # try except
            try:
                subtitle_dicts = YouTubeTranscriptApi.get_transcript(video_id)
                transcript = ' '.join( list(map(itemgetter('text'), subtitle_dicts)) )
                transcript_list.append(transcript)
                
                
                print(transcript)
                
            ## Figure out how many rows fall in here. (How many videos disabled subtitles
            except: 
                    transcript_list.append("subtitle_disabled")
            
        else: 
            transcript_list.append("none-youtube")
            
    
    return transcript_list





# Note: Following are for youtube videos which 'do not have' trasncripts (sub-title disabled)
#  So read the vidoes as 'wav' file, and extract transcript using sound recognition APIs. 

def read_video(file_name):
    print('Reading youtube wav file')
    try:
        r = sr.Recognizer()
        with sr.AudioFile(file_name) as source:
            audio = r.record(source)
        output = r.recognize_sphinx(audio)
    except IOError as exc:
        output = print('Unable to find the audio file.')
    except sr.UnknownValueError:
        output = print('Error reading audio')
    return output




options = {
'format': 'bestaudio/best', # choice of quality
'extractaudio' : True, # only keep the audio
'audioformat' : "wav", # convert to mp3
'outtmpl': './%(id)s.wav', #name
'noplaylist' : True, # download single, not playlist
}
delim = "="




def youtube_audio_transcriber(no_transciprt_youtube_links_ls):

    output_df = pd.DataFrame()

    with youtube_dl.YoutubeDL(options) as ydl:        
        
        for youtube_link in no_transciprt_youtube_links_ls:
           
            try:
                #youtube_link = 'https://www.youtube.com/watch?v=F5zgzlfnXAY&nohtml5'
                delim_pos = youtube_link.find(delim)
                youtube_id = youtube_link[delim_pos+1:delim_pos+12]
                ydl.download([youtube_link])
                x,_ = librosa.load( youtube_id+'.wav', sr=16000)
                os.remove(youtube_id+'.wav')
                sf.write(youtube_id+'.wav', x, 16000)
                wave.open(youtube_id+'.wav','r')
                transcription = read_video(youtube_id+".wav")
                print(transcription)
                # add row to output dataframe
                row_to_add = pd.Series([youtube_link,transcription])
                row_df = pd.DataFrame([row_to_add])
                output_df = pd.concat([row_df, output_df], ignore_index=True)
                os.remove(youtube_id+'.wav')
            except:
                row_to_add = pd.Series([youtube_link, "No Longer Public Video"])
                row_df = pd.DataFrame([row_to_add])
                output_df = pd.concat([row_df, output_df], ignore_index=True)
    
    output_df.columns = ['associated_youtube_link','youtube_transcript']
    return output_df