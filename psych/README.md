# Behavioral experiments

Contained in this directory are HTML files which run the behavioral tasks that we used to derive our behavioral measurements in this paper. Each HTML corresponds to a separate behavioral session. 


To reproduce our experiments, please follow the steps below: 

1. One should first host these HTMLs on one's preferred service (e.g. Amazon S3).
Note these HTMLs are static webpages; they do not require an application server to function. 
2. These HTMLs assume the user is a Worker from Amazon Mechanical Turk (AMT). To run a behavioral session in a human subject
(a HIT), an [External Question](https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/ApiReference_ExternalQuestionArticle.html) HIT should be created pointing to the URL 
containing the HTML webpage. 
3. At the end of the session, the webpage submits a POST 
request containing the user's behavioral responses to an AMT endpoint URI, which can then be retrieved using [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)  – namely the [get_assignment](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/mturk/client/get_assignment.html) method.
4. This method retrieves a JSON data file which may be processed using the methods in `process_session_data.py`, which generates an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) summarizing the worker's responses. 
5. Crucially, these tasks promise workers a monetary bonus based on their behavior. 
These bonuses may be computed from the `xarray.Dataset` object above, using the `compute_bonus_amount` function in `process_session_data.py`.   
6. These bonuses should be delivered in a timely manner to the worker, e.g. using the [`send_bonus`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/mturk/client/send_bonus.html) method in boto3.  

The HTMLs are organized experiment-wise in the zip file; see the paper for a description of all experiments run in this study. 
Note that not all HTMLs were run in the experiments described in the paper; the entire batch of HTMLs was first written, then we ran a random subset of these (the first [n] HTMLs in each experiment). We include all HTMLs for completeness, and to permit larger-scale replications of our work.
We attempted to run two subjects for each HTML included in this study. Note that we also followed specific subject recruiting and filtering procedures described in the main text. Following the conclusion of our behavioral studies, we also supplemented all workers' compensation so all would be equally compensated per HTML completed.


