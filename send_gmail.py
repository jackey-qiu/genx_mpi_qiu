#!/usr/bin/python
import smtplib
import time

try:
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()
except:
    pass


SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

 
sender = 'lingqunwu@gmail.com'
recipient = 'cqiu@alaska.edu'
subject = 'model running reminder from pacman'
body = 'Submitted runs finished. Either append new runs or kill the job. Cores will be asleep for 30 min'
 
"Sends an e-mail to the specified recipient."
 
body = "" + body + ""
 
headers = ["From: " + sender,
           "Subject: " + subject,
           "To: " + recipient,
           "MIME-Version: 1.0",
           "Content-Type: text/html"]
headers = "\r\n".join(headers)
 
session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
 
session.ehlo()
session.starttls()
session.ehlo
session.login(sender, 'wlq013550')
if rank==0:
    session.sendmail(sender, recipient, headers + "\r\n\r\n" + body)
    session.quit()
    print "email reminder sent"
else:
    pass

time.sleep(1)
