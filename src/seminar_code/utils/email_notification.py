import logging
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
# Set logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateNotification(object):
    """
    A notification is sent (via Email) when
    the a certain state of an object is reached, i.e. a specified
    training state or accuracy when machine learning models are trained.
    """
    def __init__(
        self,
        from_email: str=None,
        to_email: str=None,
        retry_count: int=5,
        credential_path: str=None,
        credential_file_name: str=None,
        smtp_server: str=None,
        smtp_port: int=None,
        smtp_password: str=None
        ) -> None:
        self.from_email = from_email
        self.to_email = to_email
        self.retry_count = retry_count
        self.credential_path = credential_path
        self.credential_file_name = credential_file_name
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port 
        self.smtp_password = smtp_password
        if (self.credential_path and self.credential_file_name) is not None:
            # trigger credential loading
            credentials = self.__load_credentials()
            self.credentials = credentials

    ######################### Internal helper methods #########################
    def __check_path_existence(
        self,
        path: str
        ):
        """Internal helper method - serves as generous path existence
           checker when saving and reading of an kind of data from files
           suspected at the given location
           
           !!!!If given path does not exist it will be created!!!!

        Args:
            path (str): full path where expected data is saved
        """
        folder_name = path.split("/")[-1]
        path = "/".join(path.split("/")[:-1])
        # FileNotFoundError()
        # os.path.isdir()
        if folder_name not in os.listdir(path):
            logging.info(f"{folder_name} not found in path: {path}")
            folder_path = f"{path}/{folder_name}"
            os.mkdir(folder_path)
            logging.info(f"Folder: {folder_name} created in path: {path}")


    def __load_credentials(
        self
        ):
        # First check if a secrets file is already present at the provided path
        if self.credential_file_name is not None and self.credential_path is not None:
            self.__check_path_existence(path=self.credential_path)
            if self.credential_file_name in os.listdir(self.credential_path):
                file_path_full = f"{self.credential_path}/{self.credential_file_name}"
                with open(file_path_full, encoding="utf-8") as json_file:
                    credentials = json.load(json_file)
                logging.info(f"Credentials loaded from file: {self.credential_file_name} in path: {self.credential_path}")
                return credentials
            else:
                raise KeyError(f"{self.credential_file_name} not found in path: {self.credential_path}")
        else:
            logging.info(f"No credentials provided, missing the file path: {self.credential_path} and/or the file name: {self.credential_file_name}")

    def send_email_notification(
        self,
        subject: str,
        body: str,
        smtp_server: str,
        smtp_port: int,
        smtp_password: str=None,
        to_email: str=None,
        from_email: str=None,
        retry_count: int=None,
        ) -> None:
        """
        Send an email notification.
        """
        # Set the default class values if nothing else is provided
        if to_email is None:
            to_email = self.to_email
        if from_email is None:
            from_email = self.from_email
        if smtp_password is None:
            smtp_password = self.smtp_password
        if retry_count is None:
            retry_count = self.retry_count
        if smtp_server is None:
            smtp_server = self.smtp_server
        if smtp_port is None:
            smtp_port = self.smtp_port
        if smtp_password is None:
            smtp_password = self.smtp_password

        # Create the email
        msg = MIMEMultipart("alternative")  # Explicitly define as multipart/alternative
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            success_send = False
            while success_send == False:
                for i in range(retry_count):
                    logging.info(f"Retry count: {i}")
                    try:
                        server = smtplib.SMTP(smtp_server, smtp_port)
                        server.starttls()  # Secure the connection
                        smtp_password_encoded = smtp_password.encode('utf-8')
                        server.login(user=from_email, password=smtp_password)
                        # Attach HTML content
                        html_part = MIMEText(body, 'html', 'utf-8')  # Set encoding to UTF-8
                        msg.attach(html_part)

                        # Send the email
                        server.sendmail(
                            from_addr=from_email,
                            to_addrs=to_email,
                            msg=msg.as_string())

                        # Close the server connection
                        server.quit()
                        success_send = True
                        logging.info(f"Email sent successfully from: {from_email} to: {to_email}")
                    except Exception as e:
                        logging.error(f"Error when trying to send Email from: {self.from_email} to: {self.to_email} with: {e}")
                break

# Example usage:
state_notification_instance = StateNotification(
    credential_path=r"/Users/Robert_Hennings/Projects/SettingsPackages",
    credential_file_name=r"credentials.json"
    )
# Set email and password
Email_SMTP_creds = state_notification_instance.credentials.get("Email_SMTP")
from_email = Email_SMTP_creds.get("Email_from")
smtp_server = Email_SMTP_creds.get("Email_SMTP_SERVER")
smtp_port = Email_SMTP_creds.get("Email_SMTP_PORT")
smtp_password = Email_SMTP_creds.get("Email_SMTP_PW")
smtp_server = Email_SMTP_creds.get("Email_SMTP_SERVER")

subject = "test"
example_body = "Hello Test"
state_notification_instance.send_email_notification(
    subject=subject,
    body=example_body,
    smtp_server=smtp_server,
    smtp_port=smtp_port,
    smtp_password=smtp_password,
    to_email=from_email,
    from_email=from_email,
)