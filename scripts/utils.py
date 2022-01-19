"""
util functions for preprocessing text.
"""
import re

def remove_emails(text):
    """
    This removes values in enclosed < >
    """
    return re.sub(r"\S*@\S*\s?", "", text)

def remove_in_article(text):
    return re.sub(r"In article.+writes:", "", text)

def remove_names(text):
    return re.sub(r"[A-Z][a-z]{1, 15}\s[A-Z][a-z]{1, 15}", "", text)

def remove_arrow(text):
    return re.sub(r">", "", text)
