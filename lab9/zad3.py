import requests
import pandas as pd
import json
import time
from datetime import datetime



SUBREDDIT = "learnprogramming"  
SORT_BY = "hot"                 
TIME_FILTER = "day"              
POST_LIMIT = 100                 
OUTPUT_PREFIX = "reddit_posts"   


def fetch_reddit_json(url, headers, params=None):
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Błąd podczas pobierania danych: {e}")
        return None


def extract_post_data(post):
    data = post['data']
    
    return {
        'id': data.get('id', ''),
        'title': data.get('title', ''),
        'author': data.get('author', '[deleted]'),
        'subreddit': data.get('subreddit', ''),
        'score': data.get('score', 0),
        'upvote_ratio': data.get('upvote_ratio', 0),
        'num_comments': data.get('num_comments', 0),
        'created_utc': datetime.fromtimestamp(data.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S'),
        'url': data.get('url', ''),
        'permalink': f"https://www.reddit.com{data.get('permalink', '')}",
        'selftext': data.get('selftext', '')[:500],  # Pierwsze 500 znaków
        'is_self': data.get('is_self', False),
        'link_flair_text': data.get('link_flair_text', ''),
        'domain': data.get('domain', ''),
        'is_video': data.get('is_video', False),
        'thumbnail': data.get('thumbnail', ''),
        'gilded': data.get('gilded', 0),
        'distinguished': data.get('distinguished', ''),
    }


def scrape_reddit_posts(subreddit, sort_by='hot', time_filter='day', limit=100):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    if sort_by == 'top':
        url = f"https://www.reddit.com/r/{subreddit}/top.json"
        params = {'t': time_filter, 'limit': 100}  # Reddit API zwraca max 100 na request
    else:
        url = f"https://www.reddit.com/r/{subreddit}/{sort_by}.json"
        params = {'limit': 100}
    
    posts_data = []
    after = None  
    
    print("="*60)
    print(f"POBIERANIE POSTÓW Z r/{subreddit}")
    print("="*60)
    print(f"Sortowanie: {sort_by}")
    if sort_by == 'top':
        print(f"Filtr czasu: {time_filter}")
    print(f"Cel: {limit} postów\n")
    
    
    while len(posts_data) < limit:
        if after:
            params['after'] = after
        
        print(f"Pobieranie... (dotychczas: {len(posts_data)} postów)")
        
        data = fetch_reddit_json(url, headers, params)
        
        if not data or 'data' not in data:
            print("Brak więcej danych do pobrania")
            break
        
        children = data['data'].get('children', [])
        
        if not children:
            print("Nie znaleziono postów")
            break
        
        for post in children:
            if post['kind'] == 't3':
                posts_data.append(extract_post_data(post))
                
                if len(posts_data) >= limit:
                    break
        
        after = data['data'].get('after')
        
        if not after:
            print("Osiągnięto koniec dostępnych postów")
            break
        
        time.sleep(1)
    
    print(f"\nPomyślnie pobrano {len(posts_data)} postów!")
    return posts_data


def save_to_files(posts, base_filename):

    if not posts:
        print("Brak postów do zapisania")
        return None, None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    csv_filename = f"{base_filename}_{timestamp}.csv"
    df = pd.DataFrame(posts)
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\nZapisano do CSV: {csv_filename}")
    
    json_filename = f"{base_filename}_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print(f"Zapisano do JSON: {json_filename}")
    
    return csv_filename, json_filename


def display_statistics(posts):
    if not posts:
        return
    
    df = pd.DataFrame(posts)
    
    print("\n" + "="*60)
    print("STATYSTYKI POBRANYCH POSTÓW")
    print("="*60)
    
    print(f"Łączna liczba postów:           {len(posts)}")
    print(f"Średnia liczba komentarzy:      {df['num_comments'].mean():.2f}")
    print(f"Średni score:                   {df['score'].mean():.2f}")
    print(f"Najwyżej oceniony post:         {df['score'].max()} punktów")
    print(f"Post z najwięcej komentarzy:    {df['num_comments'].max()} komentarzy")
    print(f"Średni upvote ratio:            {df['upvote_ratio'].mean():.2%}")
    
    top_authors = df['author'].value_counts().head(3)
    print(f"\nNajaktywniejszych autorów:")
    for i, (author, count) in enumerate(top_authors.items(), 1):
        print(f"  {i}. u/{author}: {count} postów")
    
    self_posts = df['is_self'].sum()
    link_posts = len(df) - self_posts
    print(f"\nTypy postów:")
    print(f"  Posty tekstowe:  {self_posts}")
    print(f"  Linki:           {link_posts}")


def display_sample_posts(posts, count=3):
    if not posts:
        return
    
    print("\n" + "="*60)
    print(f"PRZYKŁADOWE POSTY (pierwsze {min(count, len(posts))})")
    print("="*60)
    
    for i, post in enumerate(posts[:count], 1):
        print(f"\n{i}. {post['title']}")
        print(f"Autor: u/{post['author']}")
        print(f"Score: {post['score']} | Komentarze: {post['num_comments']}")
        print(f"Data: {post['created_utc']}")
        print(f"{post['permalink']}")
        
        if post['selftext'] and len(post['selftext']) > 0:
            preview = post['selftext'][:100] + "..." if len(post['selftext']) > 100 else post['selftext']
            print(f"Podgląd: {preview}")

def main():
    
    try:
        posts = scrape_reddit_posts(
            subreddit=SUBREDDIT,
            sort_by=SORT_BY,
            time_filter=TIME_FILTER,
            limit=POST_LIMIT
        )
        
        if posts:
            save_to_files(posts, f"{OUTPUT_PREFIX}_{SUBREDDIT}")

            display_statistics(posts)
            
            display_sample_posts(posts, 3)
            
            print("\n" + "="*60)
            print("GOTOWE!")
            print("="*60)
        else:
            print("\nNie udało się pobrać żadnych postów")
            
    except KeyboardInterrupt:
        print("\n\nPrzerwano przez użytkownika")
    except Exception as e:
        print(f"\nWystąpił nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()