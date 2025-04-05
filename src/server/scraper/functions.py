import os
from wrapt_timeout_decorator import *  # Importing timeout decorator for functions from wrapt_timeout_decorator library (not explicitly used in this file)
from linkedin_api import Linkedin  # Importing Linkedin API client
import praw  # Importing praw, the Python Reddit API Wrapper
from ntscraper import Nitter  # Importing Nitter scraper for Twitter
from dotenv import load_dotenv
from linkedin_api.cookie_repository import LinkedinSessionExpired

from .prompts import *  # Importing prompt templates and related utilities from prompts.py

load_dotenv("server/.env")  # Load environment variables from .env file
def scrape_linkedin_profile(url: str) -> dict:
    """Scrapes and formats LinkedIn profile data from a given profile URL."""
    try:
        username = os.getenv("LINKEDIN_USERNAME")
        password = os.getenv("LINKEDIN_PASSWORD")
        if not username or not password:
            raise Exception("LinkedIn credentials are missing in environment variables.")

        try:
            api = Linkedin(username, password)
        except LinkedinSessionExpired:
            raise Exception("LinkedIn session expired. Delete cached cookies or re-authenticate.")

        profile_id = url.split("/")[-2]
        profile = api.get_profile(profile_id)

        del_keys = [
            "geoCountryUrn", "geoLocationBackfilled", "elt", "industryUrn",
            "entityUrn", "geo", "urn_id", "public_id", "member_urn", "profile_urn",
            "profile_id", "img_800_800", "img_400_400", "img_200_200", "img_100_100",
            "displayPictureUrl", "geoLocation", "location", "student"
        ]

        formatted_profile = {
            k: v for k, v in profile.items() if k not in del_keys
        }

        formatted_profile["projects"] = {
            item["title"]: item["description"]
            for item in formatted_profile.get("projects", [])
        }

        formatted_profile["honors"] = {
            item["title"]: item["title"]
            for item in formatted_profile.get("honors", [])
        }

        formatted_profile["certifications"] = {
            item["name"]: item["name"]
            for item in formatted_profile.get("certifications", [])
        }

        formatted_profile["education"] = {
            item["schoolName"]: {
                "degreeName": item.get("degreeName", ""),
                "fieldOfStudy": item.get("fieldOfStudy", ""),
                "grade": item.get("grade", ""),
            }
            for item in formatted_profile.get("education", [])
        }

        formatted_profile["experience"] = {}
        for item in formatted_profile.get("experience", []):
            title = item.get("title", "")
            if title:
                formatted_profile["experience"][title] = {
                    "companyName": item.get("companyName", ""),
                    "startDate": item.get("timePeriod", {}).get("startDate", {}),
                    "endDate": item.get("timePeriod", {}).get("endDate", {}),
                    "description": item.get("description", ""),
                }

        return formatted_profile

    except Exception as e:
        raise Exception(f"Error scraping linkedin data: {e}")

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
    scraper = Nitter()

    if not scraper.working_instances:
        raise Exception("No available Nitter instances to scrape from.")

    if "http" in username_or_url:
        username = username_or_url.rstrip("/").split("/")[-1]
    else:
        username = username_or_url

    try:
        tweets_data = scraper.get_tweets(username, mode="user", number=num_tweets)
        tweet_texts = [tweet["text"] for tweet in tweets_data["tweets"]]
        return tweet_texts
    except Exception as e:
        raise Exception(f"Error scraping twitter data: {e}")