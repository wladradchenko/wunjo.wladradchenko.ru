import requests
import json


def get_translate(text: str, targetLang: str, sourceLang: str="auto") -> str:
    """
    Translate text by requests
    :param text: source text
    :param targetLang: target language
    :param sourceLang: source language
    :return: translation text
    """
    if not text:
        return text

    google_url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sourceLang}&tl={targetLang}&dt=t&q={text}"

    try:
        response = requests.get(google_url)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        translation_data = json.loads(response.text)
        return "".join(item[0] for item in translation_data[0])
    except requests.RequestException as e:
        print(f"Error... during the request to translate text: {e}")
    except (IndexError, TypeError):
        print("Error... Could not retrieve translation from response")
    except json.JSONDecodeError:
        print("Error... decoding the JSON response during translation")
    except Exception as err:
        print(f"An unexpected error during translation occurred: {err}")

    return text  # Return original text if translation fails

