"""
Fetches and downloads 20 newsgroups dataset into local files for
greater visibility
"""
import sys
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups

def inflate_newsgroups(news_groups):
    """
    writes individual text files to local folder for visibility.
    """
    for filepath, contents in zip(news_groups['filenames'], news_groups['data']):
        relative_path = list(Path(filepath).parts[-4:])
        relative_path = Path(f"{(Path(*relative_path))}.txt")
        relative_path.parent.mkdir(parents=True, exist_ok=True)
        with open(relative_path, 'w') as fh:
            fh.write(contents)
        
def inspect_news_groups(news_groups):
    print(type(news_groups))
    print(len(news_groups))
    print(type(news_groups['data']))
    print(len(news_groups['data']))


def main(output_dir):
    # downloads data into ‘~/scikit_learn_data’ directory
    # unless specify data_home parameter
    news_groups = fetch_20newsgroups(
            subset='all', # unsupervised learning, dont need test/train set
            remove=("headers", "footers") # don't want to overfit on metadata
            )
    inspect_news_groups(news_groups=news_groups)
    inflate_newsgroups(news_groups=news_groups)

if __name__ == "__main__":
    main(sys.argv[1])

