import random
from datetime import datetime, timedelta
import pandas as pd


df = pd.read_csv("C:/Users/pujar/Desktop/Project/DS/dataset/survey_data.csv")


first_names = [
    "Aarav",
    "Viren",
    "Ishaan",
    "Rajesh",
    "Kunal",
    "Deepika",
    "Ananya",
    "Pooja",
    "Meera",
    "Radhika",
    "Sandeep",
    "Arjun",
    "Neha",
    "Rohan",
    "Tanvi",
    "Harsh",
    "Aditi",
    "Saurabh",
    "Sneha",
    "Rakesh",
    "Vikram",
    "Manoj",
    "Jyoti",
    "Rekha",
    "Suhas",
    "Gopal",
    "Tushar",
    "Pankaj",
    "Siddharth",
    "Ritika",
    "Priya",
    "Rahul",
    "Madhavi",
    "Sanya",
    "Sahil",
    "Ujjwal",
    "Suresh",
    "Akash",
    "Vivek",
    "Parth",
    "Chirag",
    "Aniket",
    "Nidhi",
    "Kavya",
    "Varun",
    "Pallavi",
    "Yash",
    "Kritika",
    "Srinivas",
    "Bhavana",
]


last_names = [
    "Sharma",
    "Verma",
    "Patel",
    "Iyer",
    "Nair",
    "Reddy",
    "Das",
    "Mehta",
    "Bose",
    "Gupta",
    "Kulkarni",
    "Chatterjee",
    "Malhotra",
    "Rana",
    "Kapoor",
    "Bajaj",
    "Pillai",
    "Swamy",
    "Chopra",
    "Jain",
    "Kumar",
    "Agarwal",
    "Bhatia",
    "Singh",
    "Thakur",
    "Joshi",
    "Goswami",
    "Pandey",
    "Nanda",
    "Trivedi",
    "Mishra",
    "Dwivedi",
    "Tiwari",
    "Desai",
    "Kohli",
    "Dixit",
    "Narang",
    "Saxena",
    "Sethi",
    "Chauhan",
    "Rastogi",
    "Naidu",
    "Menon",
    "Rastogi",
    "Khanna",
    "Goyal",
    "Lalwani",
    "Bhatt",
    "Dutta",
    "Sengupta",
]

# Define possible values for each column based on observed data
age_groups = ["<18", "18 - 22", "22 - 26", ">26"]
genders = ["Male", "Female"]
locations = ["Urban", "Rural"]
mobile_usage_hours = ["<2 Hours", "2 - 4 Hours", "4 - 6 Hours", ">6 Hours"]
social_media_usage = ["Always", "Sometimes", "Rarely", "Never"]
primary_usage = ["Education", "Entertainment", "Social Media"]
educational_use = ["Yes", "No", "Maybe"]
study_time = ["<2 Hours", "2 - 4 Hours", "4 - 6 Hours", ">6 Hours"]
academic_performance = ["<50%", "50 - 70%", "70 - 85%", ">85%"]
impact_on_academics = ["Yes", "No"]
exam_usage_change = ["Increased", "Decreased", "No Change"]
tracking_apps = ["Yes", "No"]
reducing_usage = ["Yes", "No"]
distraction_or_tool = ["Tool for Learning", "Distraction"]

# Generate 50 unique and logical entries
new_entries = []
start_date = datetime(2025, 1, 28, 8, 26, 2)

end_date = datetime.now()


for i in range(50):
    time = start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    )
    first = random.choice(first_names)
    last = random.choice(last_names)
    full_name = f"{first} {last}"
    email = f"{first.lower()}{random.choice([".","_",""])}{random.choice([last.lower(),last.title(),last.capitalize()])}{random.choice(range(1000))}@gmail.com"
    age = random.choice(age_groups)
    gender = random.choice(genders)
    location = random.choice(locations)
    mobile_hours = random.choice(mobile_usage_hours)
    social_media = random.choice(social_media_usage)

    # Logical dependency: If mobile use is >6 hours, primary purpose is likely Entertainment or Work
    if mobile_hours == ">6 Hours":
        primary_purpose = random.choice(["Entertainment", "Work"])
    elif mobile_hours in ["4 - 6 Hours", "2 - 4 Hours"]:
        primary_purpose = random.choice(["Education", "Entertainment", "Communication"])
    else:
        primary_purpose = "Education"

    educational_use_response = (
        "Yes" if primary_purpose == "Education" else random.choice(educational_use)
    )

    # Logical dependency: Higher study time generally correlates with better academic performance
    if mobile_hours in [">6 Hours", "4 - 6 Hours"]:
        study_time_response = "<2 Hours"
        academic_performance_response = random.choice(["50 - 70%", "<50%"])
    else:
        study_time_response = random.choice(["2 - 4 Hours", "4 - 6 Hours"])
        academic_performance_response = random.choice(["70 - 85%", ">85%"])

    # Impact correlation
    impact_response = (
        "Yes"
        if mobile_hours in [">6 Hours", "4 - 6 Hours"]
        else random.choice(["No", "Maybe"])
    )

    # Exam preparation impact logic
    exam_change = (
        "Decreased"
        if primary_purpose == "Education"
        else random.choice(exam_usage_change)
    )

    # Tracking apps and reduction intent
    tracking_app = "Yes" if mobile_hours in [">6 Hours", "4 - 6 Hours"] else "No"
    reduce_usage = "Yes" if impact_response == "Yes" else random.choice(["No", "Maybe"])

    # Distraction vs Learning tool logic
    if primary_purpose == "Education":
        distraction_tool = "Tool for Learning"
    elif primary_purpose == "Entertainment":
        distraction_tool = "Distraction"
    else:
        distraction_tool = "Both"

    new_entries.append(
        [
            time,
            email,
            full_name,
            age,
            gender,
            location,
            mobile_hours,
            social_media,
            primary_purpose,
            educational_use_response,
            study_time_response,
            academic_performance_response,
            impact_response,
            exam_change,
            tracking_app,
            reduce_usage,
            distraction_tool,
        ]
    )
# new_entries.sort("Timestamp")

# Create a new DataFrame and display the first few rows
newdf = df.sort_values(by="Timestamp")
new_df = pd.DataFrame(newdf, columns=df.columns)
new_df.to_csv("hello.csv", index=False)
