SAVE_PATH = r"/Users/Robert_Hennings/Uni/Master/Seminar/data"
PARAMS_MAPPING_DICT={"data": ".csv?v=1&csvType=full&useColumnShortNames=true",
                     "metadata": ".metadata.json?v=1&csvType=full&useColumnShortNames=true"}
STORAGE_OPTIONS_DICT={'User-Agent': 'Our World In Data data fetch/1.0'}
REQUEST_TIMEOUT = 90
ERROR_CODES_DICT = {
        400: {
            "message": "Bad request",
            "return_": None
        },
        401: {
            "message": "Failed to authenticate, check your authenticat…",
            "return_": None
        },
        410: {
            "message": "Unauthorized",
            "return_": None
        },
        403: {
            "message": "Forbidden",
            "return_": None
        },
        404: {
            "message": "Not Found",
            "return_": None
        },
        405: {
            "message": "Method not allowed",
            "return_": None
        },
        406: {
            "message": "Not acceptable",
            "return_": None
        },
        409: {
            "message": "Conflict",
            "return_": None
        },
        415: {
            "message": "Unsopprted Media Type",
            "return_": None
        },
        500: {
            "message": "Internal Server Error",
            "return_": None
        },
        502: {
            "message": "Bad Gateway",
            "return_": None
        },
        503: {
            "message": "Service Unavailable",
            "return_": None
        },
        504: {
            "message": "Gateway Timeout",
            "return_": None
        }
    }
SUCCESS_CODES_DICT = {
    200: {
        "message": "OK",
        "return_": "Success"
        },
    201: {
    "message": "Created",
    "return_": "Success"
    },
    204: {
        "message": "No content",
        "return_": "Success"
        }
}
BIS_EXCHANGE_RATE_MAPPING_DICT = {
    "Real effective exchange rate - monthly - broad basket": "M.R.B.",
    "Real effective exchange rate - monthly - narrow basket": "M.R.N.",
    "Nominal effective exchange rate - monthly - broad basket": "M.R.B.",
    "Nominal effective exchange rate - monthly - narrow basket": "M.R.N.",
    "Nominal effective exchange rate - daily - narrow basket": "D.N.N.",
    "Nominal effective exchange rate - daily - broad basket": "D.N.B.",
}
BIS_CENTRAL_BANK_POLICY_RATE_MAPPING_DICT = {
    "Central bank policy rate - daily": "D."
}
BIS_DATA_BASE_URL = "https://stats.bis.org/api/v2/"