"""
Module that makes use of the Spotify Web API to retrieve pseudo-random songs based
or not on a given exiting Spotify genre (look at genres.json, filled with info
scrapped from http://everynoise.com/everynoise1d.cgi?scope=all&vector=popularity)
Spotify Ref: https://developer.spotify.com/documentation/web-api/reference-beta/#category-search
"""
import base64
import json
import random
import requests

# Client Keys
CLIENT_ID = "YOUR CLIENT ID"
CLIENT_SECRET = "YOUR CLIENT SECRET"

# Spotify API URIs
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com"
API_VERSION = "v1"
SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)


def get_token():
    client_token = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode('UTF-8')).decode('ascii')
    headers = {"Authorization": f"Basic {client_token}"}
    payload = {"grant_type": "client_credentials"}
    token_request = requests.post(SPOTIFY_TOKEN_URL, data=payload, headers=headers)
    access_token = json.loads(token_request.text)["access_token"]
    return access_token


def request_valid_song(access_token):
    # Wildcards for random search
    random_wildcards = ['%25a%25', 'a%25', '%25a',
                        '%25e%25', 'e%25', '%25e',
                        '%25i%25', 'i%25', '%25i',
                        '%25o%25', 'o%25', '%25o',
                        '%25u%25', 'u%25', '%25u']
    wildcard = random.choice(random_wildcards)

    # Make a request for the Search API with pattern and random index
    authorization_header = {"Authorization": f"Bearer {access_token}"}

    # Default value in case we don't find any song.
    artist = "Rick Astley"
    song = "Never Gonna Give You Up"
    for i in range(51):
        print(wildcard)
        try:
            song_request = requests.get(
                f'{SPOTIFY_API_URL}/search?q={wildcard}%20genre:%22%22&type=track&offset={random.randint(0, 200)}',
                headers=authorization_header
            )
            song_info = random.choice(json.loads(song_request.text)['tracks']['items'])
            artist = song_info['artists'][0]['name']
            song = song_info['name']
            break
        except IndexError:
            continue

    return f"{artist} - {song}"


def get_random_song():
    # Default value in case we don't find any song.
    artist = "Rick Astley"
    song = "Never Gonna Give You Up"
    from string import digits, ascii_uppercase, ascii_lowercase

    chars = digits + ascii_uppercase + ascii_lowercase
    for i in range(51):
        try:
            key = "".join(random.choice(chars) for _ in range(22))
            print(key)
            song_request = requests.get(
                f'https://open.spotify.com/track/{key}'
            )
            print(song_request.text)
            if "<!doctype" != song_request.text:
                break
        except IndexError:
            continue

    return f"{artist} - {song}"


def main():
    # Get a Spotify API token
    # access_token = get_token()

    # Call the API for a song that matches the criteria
    # result = request_valid_song(access_token)
    result = get_random_song()
    print(result)


if __name__ == '__main__':
    main()
