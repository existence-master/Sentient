import os
from wrapt_timeout_decorator import *  # Importing timeout decorator for functions from wrapt_timeout_decorator library (not explicitly used in this file)
from linkedin_api import Linkedin  # Importing Linkedin API client
import praw  # Importing praw, the Python Reddit API Wrapper
from ntscraper import Nitter  # Importing Nitter scraper for Twitter
from dotenv import load_dotenv

from .prompts import *  # Importing prompt templates and related utilities from prompts.py

load_dotenv("model/.env")  # Load environment variables from .env file

def scrape_linkedin_profile(url: str) -> dict:
    """
    Scrapes and formats LinkedIn profile data from a given profile URL.

    Utilizes the Linkedin API client to fetch profile information, then processes and formats
    the data, removing unnecessary keys and restructuring certain sections like 'projects', 'honors',
    'certifications', 'education', and 'experience' for better readability and use.

    Args:
        url (str): The LinkedIn profile URL to scrape.

    Returns:
        dict: A formatted dictionary containing relevant LinkedIn profile information.
              Sections like 'projects', 'honors', 'certifications', 'education', and 'experience'
              are restructured into more accessible formats.

    Raises:
        Exception: If there is an error during LinkedIn data scraping or API interaction.
    """
    try:
        api = Linkedin(
            os.getenv("LINKEDIN_USERNAME"), os.getenv("LINKEDIN_PASSWORD")
        )  # Initialize LinkedIn API client
        
        profile_id = url.split("/")[-2]  # Extract profile ID from URL
        profile = api.get_profile(profile_id)  # Fetch profile data using API
        
        # Keys to delete from the profile data, considered unnecessary for the application
        del_keys = [
            "geoCountryUrn",
            "geoLocationBackfilled",
            "elt",
            "industryUrn",
            "entityUrn",
            "geo",
            "urn_id",
            "public_id",
            "member_urn",
            "profile_urn",
            "profile_id",
            "img_800_800",
            "img_400_400",
            "img_200_200",
            "img_100_100",
            "displayPictureUrl",
            "geoLocation",
            "location",
            "student",
        ]

        formatted_profile = {}  # Initialize dictionary to store formatted profile data

        for item in profile.items():  # Iterate through profile items
            if item[0] not in del_keys:
                formatted_profile[item[0]] = item[
                    1
                ]  # Keep item if key is not in del_keys

        # Restructure 'projects' section
        projects = {}
        for item in formatted_profile["projects"]:
            projects[item["title"]] = item[
                "description"
            ]  # Map project title to description
        formatted_profile["projects"] = projects

        # Restructure 'honors' section
        honors = {}
        for item in formatted_profile["honors"]:
            honors[item["title"]] = item[
                "title"
            ]  # Map honor title to itself (for simple listing)
        formatted_profile["honors"] = honors

        # Restructure 'certifications' section
        certifications = {}
        for item in formatted_profile["certifications"]:
            certifications[item["name"]] = item[
                "name"
            ]  # Map certification name to itself (for simple listing)
        formatted_profile["certifications"] = certifications

        # Restructure 'education' section
        education = {}
        for item in formatted_profile["education"]:
            education[item["schoolName"]] = {
                "degreeName": item.get(
                    "degreeName", ""
                ),  # Get degree name, default to empty string if not present
                "fieldOfStudy": item.get(
                    "fieldOfStudy", ""
                ),  # Get field of study, default to empty string if not present
                "grade": item.get(
                    "grade", ""
                ),  # Get grade, default to empty string if not present
            }
        formatted_profile["education"] = education

        # Restructure 'experience' section
        experience = {}
        for item in formatted_profile["experience"]:
            title = item.get(
                "title", ""
            )  # Get job title, default to empty string if not present
            if title:  # Process only if title is present
                experience[title] = {
                    "companyName": item.get(
                        "companyName", ""
                    ),  # Get company name, default to empty string if not present
                    "startDate": item["timePeriod"]["startDate"]
                    if "timePeriod" in item and "startDate" in item["timePeriod"]
                    else {},  # Get start date, handle missing keys
                    "endDate": item["timePeriod"]["endDate"]
                    if "timePeriod" in item and "endDate" in item["timePeriod"]
                    else {},  # Get end date, handle missing keys
                    "description": item.get(
                        "description", ""
                    ),  # Get description, default to empty string if not present
                }
        formatted_profile["experience"] = experience

        return formatted_profile  # Return the formatted LinkedIn profile data
    except Exception as e:
        print(f"Error scraping LinkedIn data: {str(e)}")
        raise Exception(
            f"Error scraping linkedin data: {e}"
        )  # Re-raise exception with more context


def reddit_scraper(url: str, limit: int = 50) -> dict:
    """
    Scrape subreddit data for a Reddit user, fetching submissions and comments.

    Utilizes praw (Python Reddit API Wrapper) to access Reddit data. It fetches the latest submissions
    and comments for a given user, up to a specified limit, and extracts the subreddits they interacted with.

    Args:
        url (str): The URL of the Reddit user profile to scrape.
        limit (int, optional): The maximum number of submissions and comments to fetch. Defaults to 50.

    Returns:
        dict: A dictionary containing sets of subreddits where the user has posted submissions and comments.
              Structure: {"submissions": list[str], "comments": list[str]}

    Raises:
        Exception: If there is an error during Reddit data scraping or API interaction.
    """
    try:
        username = url.strip("/").split("/")[-1]  # Extract username from URL

        reddit = praw.Reddit(
            client_id="2RdYRgD1GmpjRQcnjwNRyA",  # Reddit API client ID
            client_secret="9pup4fJmjYSp3XdIBt-WgFyWswP1Ng",  # Reddit API client secret
            user_agent=f"myapp:v0.0.1 (by /u/{username})",  # Reddit API user agent string
        )
        user = reddit.redditor(username)  # Get Reddit user object

        subreddit_dict = {
            "submissions": set(),
            "comments": set(),
        }  # Initialize dictionary to store subreddits

        for submission in user.submissions.new(limit=limit):  # Fetch latest submissions
            subreddit_dict["submissions"].add(
                submission.subreddit.display_name
            )  # Add subreddit display name to submissions set

        for comment in user.comments.new(limit=limit):  # Fetch latest comments
            subreddit_dict["comments"].add(
                comment.subreddit.display_name
            )  # Add subreddit display name to comments set

        subreddit_dict["submissions"] = list(
            subreddit_dict["submissions"]
        )  # Convert submissions set to list
        subreddit_dict["comments"] = list(
            subreddit_dict["comments"]
        )  # Convert comments set to list

        return subreddit_dict  # Return dictionary of subreddits for submissions and comments
    except Exception as e:
        print(f"Error creating chat runnable: {e}")
        raise Exception(
            f"Error scraping reddit data: {e}"
        )  # Re-raise exception with more context


def scrape_twitter_data(username_or_url: str, num_tweets: int = 20) -> list[str]:
    """
    Scrapes tweets from a Twitter user profile and returns the text content of the latest tweets.

    Uses Nitter scraper to fetch tweets, as the official Twitter API (or tweepy) is not used here.
    It extracts the text from the specified number of latest tweets from a given username or profile URL.

    Args:
        username_or_url (str): Twitter username or profile URL to scrape tweets from.
        num_tweets (int, optional): Number of tweets to scrape. Defaults to 20.

    Returns:
        list[str]: A list of strings, where each string is the text content of a scraped tweet.

    Raises:
        Exception: If there is an error during Twitter data scraping using Nitter.
    """
    scraper = Nitter()  # Initialize Nitter scraper

    if "http" in username_or_url:
        username = username_or_url.split("/")[
            -1
        ]  # Extract username from URL if URL is provided
    else:
        username = username_or_url  # Use provided username directly

    try:
        tweets_data = scraper.get_tweets(
            username, mode="user", number=num_tweets
        )  # Scrape tweets using Nitter
        tweet_texts = [
            tweet["text"] for tweet in tweets_data["tweets"]
        ]  # Extract text from each tweet in the scraped data
        return tweet_texts  # Return list of tweet texts
    except Exception as e:
        print(f"Error creating chat runnable: {e}")
        raise Exception(
            f"Error scraping twitter data: {e}"
        )  # Re-raise exception with more context
