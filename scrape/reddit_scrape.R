# for extracting reddit posts
library(RedditExtractoR)
library(tidyverse)

# ===============================================================================
# Scraping the sub-reddits 
# ===============================================================================

# INITIAL CODE TO SCRAPE THE REDDIT DATA USING REDDITEXTRACTOR, 

climate_change <- find_thread_urls(
  subreddit = "climatechange", sort_by = "new", period = "day")

nba <- find_thread_urls(
  subreddit = 'nba', sort_by = 'new', period = 'day')

guns <- find_thread_urls(
  subreddit = 'guns', sort_by = "new", period = 'day')

repub <- find_thread_urls(
  subreddit = 'Republican', sort_by = "new", period = 'day')

dems <- find_thread_urls(
  subreddit = 'democrats', sort_by = "new", period = 'day')

snl <- find_thread_urls(
  subreddit = 'saturdaynightlive', sort_by = "new", period = 'day')

abortion <- find_thread_urls(
  subreddit = 'abortion', sort_by = "new", period = 'day')


# save to each college's posts to one r data file
all_posts <- bind_rows(climate_change, nba, guns, repub, dems, snl, abortion)



write.csv(x = all_posts, file = "scraped_posts.csv", 
          fileEncoding = "UTF-8", row.names = FALSE)