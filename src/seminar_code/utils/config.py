BASE_URL = "https://www.epexspot.com/en/market-results"
ERROR_CODES_DICT = {
        400: {
            "message": "Error",
            "return_": None
        },
        401: {
            "message": "Error",
            "return_": None
        }
    }
SUCCESS_CODES_DICT = {
    200: {
        "message": "URL succesfully retrived",
        "return_": "Success"
        }
}