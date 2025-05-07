import string

CHAR_TYPES = {
    "number": string.digits,
    "letter": string.ascii_letters,
    "letterL": string.ascii_lowercase,
    "letterU": string.ascii_uppercase,
    "mix": string.digits + string.ascii_letters,
    "mixL": string.digits + string.ascii_lowercase,
    "mixU": string.digits + string.ascii_uppercase,
}
