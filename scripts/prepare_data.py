"""
Data Preparation Script — Generates job descriptions dataset and prepares resume data.
Uses real Kaggle resume dataset structure + curated job descriptions.
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_DIR, JOB_DESC_CSV, RESUMES_CSV, SKILLS_DB_JSON, JOB_CATEGORIES

random.seed(42)
np.random.seed(42)


# ── Job Description Templates (Based on Real-World Postings) ──────────────

JOB_TEMPLATES = {
    "Data Science": [
        {
            "title": "Data Scientist",
            "description": "We are looking for a Data Scientist to analyze large amounts of raw information to find patterns and build data-driven solutions. You will be responsible for building machine learning models, performing statistical analysis, and creating data visualization dashboards. The ideal candidate has strong experience with Python, SQL, and machine learning frameworks. You will work with cross-functional teams to identify business opportunities and deliver actionable insights through advanced analytics. Experience with deep learning, NLP, and cloud platforms is a plus.",
            "required_skills": "Python,SQL,Machine Learning,Pandas,NumPy,Scikit-learn,TensorFlow,Statistical Analysis,Data Visualization,Tableau,Deep Learning,Feature Engineering",
            "experience_level": "Mid Level"
        },
        {
            "title": "Senior Data Scientist",
            "description": "As a Senior Data Scientist, you will lead complex analytics projects from inception to deployment. You will design and implement machine learning pipelines, mentor junior team members, and collaborate with product teams to drive data-informed decision making. Strong expertise in Python, advanced statistics, and production ML systems is required. Experience with A/B testing, causal inference, and deploying models at scale using cloud services (AWS/GCP) is essential. You will present findings to senior leadership and influence strategic direction.",
            "required_skills": "Python,SQL,Machine Learning,Deep Learning,TensorFlow,PyTorch,Apache Spark,AWS,A/B Testing,Statistical Analysis,Data Visualization,Leadership,Communication",
            "experience_level": "Senior"
        },
        {
            "title": "Junior Data Analyst",
            "description": "Entry-level position for a Data Analyst to support our analytics team. You will assist in data collection, cleaning, and analysis using SQL and Python. Create reports and dashboards using Tableau or Power BI. The ideal candidate has basic knowledge of statistics, Excel, and data visualization. Strong attention to detail and willingness to learn are essential.",
            "required_skills": "SQL,Python,Excel,Tableau,Statistical Analysis,Data Visualization,Pandas,Communication",
            "experience_level": "Entry Level"
        },
    ],
    "Python Developer": [
        {
            "title": "Python Backend Developer",
            "description": "We are seeking a Python Backend Developer to build scalable server-side applications. You will design and implement RESTful APIs using Django or Flask, work with PostgreSQL databases, and deploy applications using Docker and AWS. Strong understanding of software design patterns, testing, and version control with Git is required. Experience with asynchronous programming, message queues, and microservices architecture is preferred.",
            "required_skills": "Python,Django,Flask,REST API,PostgreSQL,Docker,AWS,Git,SQL,Unit Testing,Redis,Linux",
            "experience_level": "Mid Level"
        },
        {
            "title": "Senior Python Developer",
            "description": "Lead Python development for high-performance applications. Design system architecture, implement complex business logic, and ensure code quality through reviews and testing. Must have strong experience with Django/FastAPI, PostgreSQL, Docker, Kubernetes, and CI/CD pipelines. Experience with microservices, event-driven architecture, and cloud-native development is essential.",
            "required_skills": "Python,Django,FastAPI,PostgreSQL,Docker,Kubernetes,CI/CD,AWS,REST API,Git,Redis,Apache Kafka,Agile,Leadership",
            "experience_level": "Senior"
        },
    ],
    "Java Developer": [
        {
            "title": "Java Software Engineer",
            "description": "Looking for a Java Software Engineer to develop enterprise-grade applications using Spring Boot ecosystem. You will build microservices, implement REST APIs, and work with relational databases. Strong knowledge of Java 11+, Spring Framework, Hibernate, and Maven/Gradle is required. Experience with Kafka, Docker, and cloud platforms is a plus.",
            "required_skills": "Java,Spring Boot,Hibernate,Maven,REST API,SQL,PostgreSQL,Docker,Git,JUnit,Microservices,Agile",
            "experience_level": "Mid Level"
        },
        {
            "title": "Senior Java Developer",
            "description": "Lead the design and development of scalable Java applications. You will architect microservices solutions, mentor developers, and drive technical decision-making. Expert knowledge of Spring Boot, Kafka, Kubernetes, and cloud platforms (AWS/Azure) is required. Experience with performance optimization, distributed systems, and DevOps practices is essential.",
            "required_skills": "Java,Spring Boot,Spring Security,Hibernate,Apache Kafka,Kubernetes,AWS,Docker,PostgreSQL,Oracle,Redis,Microservices,REST API,Leadership",
            "experience_level": "Senior"
        },
    ],
    "Web Designing": [
        {
            "title": "Frontend Developer",
            "description": "We need a creative Frontend Developer to build responsive, user-friendly web interfaces. You will work with React/Angular, implement pixel-perfect designs from Figma mockups, and optimize performance. Strong knowledge of HTML5, CSS3, JavaScript, and modern frontend frameworks is required. Experience with TypeScript, Tailwind CSS, and responsive design is preferred.",
            "required_skills": "HTML,CSS,JavaScript,React,TypeScript,Tailwind CSS,Figma,Git,REST API,Responsive Design,Sass,Webpack",
            "experience_level": "Mid Level"
        },
        {
            "title": "Full Stack Web Developer",
            "description": "Build complete web applications from frontend to backend. Work with React/Next.js on the frontend and Node.js/Express on the backend. Manage MongoDB/PostgreSQL databases and deploy to cloud platforms. Strong knowledge of the entire web development stack, REST APIs, and version control is required.",
            "required_skills": "HTML,CSS,JavaScript,React,Node.js,MongoDB,PostgreSQL,REST API,Git,Docker,TypeScript,Next.js,Tailwind CSS",
            "experience_level": "Mid Level"
        },
    ],
    "DevOps Engineer": [
        {
            "title": "DevOps Engineer",
            "description": "Looking for a DevOps Engineer to build and maintain CI/CD pipelines, manage cloud infrastructure, and improve system reliability. You will work with Docker, Kubernetes, Terraform, and AWS services. Strong Linux administration skills and scripting ability (Python/Bash) are required. Experience with monitoring tools (Prometheus, Grafana) and GitOps practices is preferred.",
            "required_skills": "Docker,Kubernetes,AWS,Terraform,Jenkins,CI/CD,Linux,Python,Shell Scripting,Git,Prometheus,Grafana,Ansible,Nginx",
            "experience_level": "Mid Level"
        },
        {
            "title": "Senior DevOps Engineer",
            "description": "Lead infrastructure and DevOps initiatives for large-scale systems. Design highly available architectures, implement security best practices, and drive automation culture. Expert knowledge of Kubernetes, multi-cloud environments, and infrastructure as code is required. You will manage incident response, capacity planning, and disaster recovery strategies.",
            "required_skills": "Docker,Kubernetes,AWS,Azure,GCP,Terraform,Ansible,CI/CD,Linux,Python,Prometheus,Grafana,Helm,Security,Leadership",
            "experience_level": "Senior"
        },
    ],
    "Business Analyst": [
        {
            "title": "Business Analyst",
            "description": "We are looking for a Business Analyst to bridge the gap between business stakeholders and technology teams. You will gather requirements, create detailed documentation, and analyze business processes. Strong skills in SQL, Excel, and visualization tools (Tableau/Power BI) are required. Experience with Agile methodology and JIRA is preferred.",
            "required_skills": "SQL,Excel,Tableau,Power BI,JIRA,Communication,Requirements Gathering,Agile,Stakeholder Management,Problem Solving,Data Visualization",
            "experience_level": "Mid Level"
        },
    ],
    "Full Stack Developer": [
        {
            "title": "Full Stack Developer",
            "description": "Looking for a Full Stack Developer proficient in both frontend and backend technologies. You will build complete web applications using React/Angular on the frontend and Node.js/Python on the backend. Experience with databases (PostgreSQL, MongoDB), Docker, and cloud deployment is required. Strong understanding of REST APIs, version control, and agile methodologies is essential.",
            "required_skills": "JavaScript,React,Node.js,Python,PostgreSQL,MongoDB,Docker,REST API,Git,HTML,CSS,TypeScript,AWS,Agile",
            "experience_level": "Mid Level"
        },
    ],
    "Network Security Engineer": [
        {
            "title": "Cybersecurity Analyst",
            "description": "We are seeking a Cybersecurity Analyst to protect our systems and data from security threats. You will conduct vulnerability assessments, monitor security events using SIEM tools, and implement security policies. Knowledge of network security, penetration testing, and incident response is required. Certifications such as CISSP, CEH, or CompTIA Security+ are preferred.",
            "required_skills": "Cybersecurity,Penetration Testing,Firewall,SIEM,Linux,Python,Networking,Encryption,OWASP,Wireshark,IAM,Nmap,SOC",
            "experience_level": "Mid Level"
        },
    ],
    "Database Administrator": [
        {
            "title": "Database Administrator",
            "description": "Manage and optimize database systems to ensure high performance, availability, and security. You will work with PostgreSQL, MySQL, and Oracle databases, perform backup and recovery operations, and optimize queries. Experience with database monitoring, replication, and cloud database services (AWS RDS, Azure SQL) is required.",
            "required_skills": "PostgreSQL,MySQL,Oracle,SQL,Database,Linux,Python,AWS,Backup,Performance Tuning,Monitoring,Shell Scripting",
            "experience_level": "Mid Level"
        },
    ],
    "Automation Testing": [
        {
            "title": "QA Automation Engineer",
            "description": "Build and maintain automated test frameworks for web and mobile applications. You will write test scripts using Selenium, design test strategies, and integrate tests into CI/CD pipelines. Strong knowledge of testing methodologies, Java/Python, and API testing is required. Experience with Cypress, Appium, and performance testing tools is a plus.",
            "required_skills": "Selenium,Java,Python,Test Automation,API Testing,CI/CD,Git,JIRA,Agile,JUnit,TestNG,REST API,BDD,Performance Testing",
            "experience_level": "Mid Level"
        },
    ],
    "Hadoop Developer": [
        {
            "title": "Big Data Engineer",
            "description": "Design and implement big data solutions using Hadoop ecosystem. You will build data pipelines using Spark, Hive, and Kafka, process petabyte-scale datasets, and optimize data workflows. Strong knowledge of distributed computing, SQL, and Python/Scala is required. Experience with cloud data platforms (Snowflake, Databricks) is preferred.",
            "required_skills": "Hadoop,Apache Spark,Hive,Kafka,Python,SQL,Scala,MapReduce,HBase,Data Warehousing,ETL,AWS,Linux",
            "experience_level": "Mid Level"
        },
    ],
    "ETL Developer": [
        {
            "title": "ETL Developer",
            "description": "Build and maintain ETL pipelines for data integration and warehousing. You will extract data from multiple sources, transform it according to business rules, and load it into data warehouses. Experience with ETL tools (Informatica, Talend, SSIS), SQL, and Python is required. Knowledge of data modeling and cloud data platforms is preferred.",
            "required_skills": "ETL,SQL,Python,Informatica,Data Warehousing,Apache Spark,Airflow,PostgreSQL,AWS,Shell Scripting,Snowflake",
            "experience_level": "Mid Level"
        },
    ],
    "DotNet Developer": [
        {
            "title": ".NET Developer",
            "description": "Develop enterprise applications using .NET framework and C#. You will build web APIs using ASP.NET Core, work with SQL Server databases, and implement microservices architecture. Strong knowledge of C#, .NET Core, Entity Framework, and Azure is required. Experience with CI/CD, Docker, and agile methodologies is preferred.",
            "required_skills": "C#,.NET,ASP.NET,SQL Server,Azure,REST API,Entity Framework,Git,Docker,Agile,Unit Testing,JavaScript,HTML,CSS",
            "experience_level": "Mid Level"
        },
    ],
    "Blockchain Developer": [
        {
            "title": "Blockchain Developer",
            "description": "Build decentralized applications and smart contracts on Ethereum and other blockchain platforms. You will write Solidity smart contracts, develop Web3 interfaces, and implement DeFi protocols. Strong knowledge of blockchain fundamentals, cryptography, and JavaScript/TypeScript is required. Experience with Hardhat, Truffle, and IPFS is preferred.",
            "required_skills": "Blockchain,Solidity,Ethereum,Web3,JavaScript,Smart Contracts,DeFi,Python,Git,Cryptography,Node.js,TypeScript",
            "experience_level": "Mid Level"
        },
    ],
    "Testing": [
        {
            "title": "Software Test Engineer",
            "description": "Ensure software quality through comprehensive manual and automated testing. You will create test plans, execute test cases, report defects, and collaborate with development teams. Knowledge of testing methodologies, SDLC, and defect tracking tools is required. Experience with Selenium, API testing, and Agile practices is preferred.",
            "required_skills": "Manual Testing,Test Automation,Selenium,API Testing,JIRA,SQL,Agile,Unit Testing,Performance Testing,Communication,Problem Solving",
            "experience_level": "Mid Level"
        },
    ],
    "SAP Developer": [
        {
            "title": "SAP Technical Consultant",
            "description": "Configure and customize SAP modules to meet business requirements. You will develop ABAP programs, create reports, and support SAP implementations. Strong knowledge of SAP ERP, ABAP programming, and database concepts is required. Experience with SAP HANA, Fiori, and integration technologies is preferred.",
            "required_skills": "SAP,Oracle,SQL,ABAP,SAP HANA,Communication,Problem Solving,Project Management,Agile",
            "experience_level": "Mid Level"
        },
    ],
    "Mechanical Engineer": [
        {
            "title": "Mechanical Design Engineer",
            "description": "Design and develop mechanical systems and components using CAD software. You will perform stress analysis, create technical drawings, and collaborate with manufacturing teams. Knowledge of SolidWorks, AutoCAD, and engineering fundamentals is required. Experience with FEA, MATLAB, and product development lifecycle is preferred.",
            "required_skills": "AutoCAD,SolidWorks,MATLAB,C++,Problem Solving,Project Management,Communication,Teamwork",
            "experience_level": "Mid Level"
        },
    ],
    "Electrical Engineering": [
        {
            "title": "Electrical Design Engineer",
            "description": "Design electrical systems, circuits, and control systems for industrial applications. You will create schematics, perform simulations, and oversee installations. Knowledge of AutoCAD Electrical, MATLAB/Simulink, and power systems is required. Experience with PLC programming and embedded systems is a plus.",
            "required_skills": "AutoCAD,MATLAB,C++,Python,Problem Solving,Communication,Project Management,Linux",
            "experience_level": "Mid Level"
        },
    ],
    "Civil Engineer": [
        {
            "title": "Civil/Structural Engineer",
            "description": "Design and oversee construction projects including buildings, bridges, and infrastructure. You will prepare structural calculations, review designs, and ensure compliance with building codes. Knowledge of AutoCAD, structural analysis software, and project management is required.",
            "required_skills": "AutoCAD,Project Management,Communication,Problem Solving,Excel,Leadership,Teamwork",
            "experience_level": "Mid Level"
        },
    ],
    "HR": [
        {
            "title": "HR Business Partner",
            "description": "Partner with business leaders to drive HR strategy, talent management, and employee engagement. You will manage recruitment, performance reviews, and employee relations. Strong communication, organizational, and analytical skills are required. Experience with HRIS systems and employment law is preferred.",
            "required_skills": "Communication,Leadership,Excel,Presentation,Negotiation,Problem Solving,Stakeholder Management,Project Management",
            "experience_level": "Mid Level"
        },
    ],
    "Sales": [
        {
            "title": "Business Development Manager",
            "description": "Drive revenue growth through strategic sales initiatives and client relationship management. You will identify new business opportunities, negotiate contracts, and manage key accounts. Strong communication, negotiation, and presentation skills are required. Experience with CRM tools (Salesforce) and B2B sales is preferred.",
            "required_skills": "Communication,Negotiation,Presentation,Salesforce,Excel,Leadership,Problem Solving,Stakeholder Management,Time Management",
            "experience_level": "Mid Level"
        },
    ],
    "Operations Manager": [
        {
            "title": "Operations Manager",
            "description": "Oversee daily operations, optimize processes, and manage team performance. You will implement operational strategies, monitor KPIs, and drive continuous improvement. Strong leadership, analytical, and project management skills are required. Experience with Lean/Six Sigma and ERP systems is preferred.",
            "required_skills": "Leadership,Project Management,Communication,Excel,Problem Solving,Agile,Stakeholder Management,Time Management,Presentation",
            "experience_level": "Senior"
        },
    ],
    "Advocate": [
        {
            "title": "Corporate Lawyer",
            "description": "Provide legal counsel on corporate matters including contracts, compliance, and intellectual property. You will draft legal documents, review agreements, and represent the company in legal proceedings. Strong analytical, communication, and research skills are required.",
            "required_skills": "Communication,Negotiation,Problem Solving,Leadership,Presentation,Time Management",
            "experience_level": "Mid Level"
        },
    ],
    "Arts": [
        {
            "title": "Creative Director",
            "description": "Lead creative vision and design strategy for brands and campaigns. You will oversee design teams, direct visual content creation, and ensure brand consistency. Strong skills in Adobe Creative Suite, design thinking, and team management are required.",
            "required_skills": "Photoshop,Illustrator,Figma,Communication,Leadership,Presentation,Teamwork,Problem Solving",
            "experience_level": "Senior"
        },
    ],
    "Health and Fitness": [
        {
            "title": "Health Data Analyst",
            "description": "Analyze health and fitness data to drive insights for wellness programs. You will work with medical datasets, perform statistical analysis, and create reports. Knowledge of data analysis tools, statistics, and health informatics is required.",
            "required_skills": "Python,SQL,Excel,Statistical Analysis,Data Visualization,Communication,Problem Solving,Tableau",
            "experience_level": "Mid Level"
        },
    ],
}


def generate_job_descriptions():
    """Generate comprehensive job descriptions dataset."""
    jobs = []
    job_id = 1

    for category, templates in JOB_TEMPLATES.items():
        for template in templates:
            # Add the base template
            jobs.append({
                "job_id": job_id,
                "title": template["title"],
                "category": category,
                "description": template["description"],
                "required_skills": template["required_skills"],
                "experience_level": template["experience_level"],
            })
            job_id += 1

            # Create variations
            for var in range(2):
                variation = template.copy()
                variation["job_id"] = job_id
                levels = ["Entry Level", "Mid Level", "Senior", "Lead"]
                variation["experience_level"] = random.choice(levels)

                # Slightly modify title
                prefixes = ["", "Senior ", "Lead ", "Junior ", "Staff "]
                prefix = random.choice(prefixes)
                base_title = template["title"].replace("Senior ", "").replace("Junior ", "").replace("Lead ", "")
                variation["title"] = f"{prefix}{base_title}".strip()

                # Modify description slightly
                additions = [
                    " Remote-friendly position with competitive benefits.",
                    " Join our fast-growing team in this exciting role.",
                    " Great opportunity for career growth and learning.",
                    " Work with cutting-edge technologies in a collaborative environment.",
                    " Hybrid work model with flexible hours available.",
                ]
                variation["description"] = template["description"] + random.choice(additions)

                jobs.append(variation)
                job_id += 1

    df = pd.DataFrame(jobs)
    df.to_csv(JOB_DESC_CSV, index=False, encoding='utf-8')
    print(f"✅ Generated {len(df)} job descriptions across {len(JOB_TEMPLATES)} categories")
    print(f"   Categories: {df['category'].nunique()}")
    print(f"   Saved to: {JOB_DESC_CSV}")
    return df


def generate_resume_dataset():
    """
    Generate resume dataset based on Kaggle UpdatedResumeDataSet structure.
    Uses realistic resume text patterns for each category.
    """
    resume_templates = {
        "Data Science": [
            "Experienced data scientist with expertise in Python, machine learning, and statistical analysis. Built predictive models using scikit-learn, TensorFlow, and XGBoost. Proficient in SQL, Pandas, and data visualization with Tableau. Conducted A/B testing and implemented recommendation systems. Experience with NLP, deep learning, and cloud platforms like AWS SageMaker.",
            "Data science professional specializing in natural language processing and computer vision. Developed end-to-end ML pipelines using PyTorch and Hugging Face transformers. Strong background in statistics, feature engineering, and model deployment. Experienced with Apache Spark for big data processing and Airflow for workflow orchestration.",
            "Results-driven data scientist with 5+ years of experience in predictive modeling and business analytics. Expert in Python, R, SQL, and machine learning algorithms. Created dashboards using Power BI and Plotly. Implemented deep learning models for image classification and text analysis. AWS certified with experience in deploying models at scale.",
        ],
        "Python Developer": [
            "Python developer with strong experience in Django, Flask, and FastAPI frameworks. Built RESTful APIs serving millions of requests. Proficient in PostgreSQL, Redis, and Docker. Implemented CI/CD pipelines and automated testing. Experience with microservices architecture and cloud deployment on AWS.",
            "Backend Python developer specializing in scalable web applications. Expert in Django REST Framework, Celery, and RabbitMQ. Strong database design skills with PostgreSQL and MongoDB. Experience with Docker, Kubernetes, and AWS services. Passionate about clean code and test-driven development.",
        ],
        "Java Developer": [
            "Java developer with 5+ years of experience in Spring Boot and microservices. Built enterprise applications using Spring MVC, Hibernate, and JPA. Proficient in Apache Kafka, Redis, and Oracle databases. Experience with Docker, Kubernetes, and CI/CD pipelines using Jenkins.",
            "Senior Java engineer specializing in high-performance distributed systems. Expert in Spring ecosystem, Maven, and JUnit testing. Built event-driven architectures using Kafka and RabbitMQ. Experience with AWS, Docker, and Kubernetes for cloud-native deployments.",
        ],
        "Web Designing": [
            "Frontend developer skilled in React, Angular, and Vue.js. Created responsive web applications using HTML5, CSS3, and JavaScript. Proficient in TypeScript, Tailwind CSS, and modern build tools. Experience with RESTful APIs, GraphQL, and state management libraries.",
            "Creative web developer with expertise in modern JavaScript frameworks. Built progressive web applications using React and Next.js. Strong UI/UX sense with experience in Figma design. Proficient in CSS animations, responsive design, and web performance optimization.",
        ],
        "DevOps Engineer": [
            "DevOps engineer with expertise in CI/CD, Docker, and Kubernetes. Managed AWS infrastructure using Terraform and CloudFormation. Implemented monitoring solutions with Prometheus and Grafana. Strong Linux administration and Python scripting skills.",
            "Cloud infrastructure engineer specializing in Kubernetes orchestration and GitOps practices. Experience with multi-cloud environments (AWS, Azure, GCP). Built automated deployment pipelines using Jenkins and ArgoCD. Expert in infrastructure as code with Terraform and Ansible.",
        ],
        "Business Analyst": [
            "Business analyst with strong analytical skills in SQL, Excel, and Tableau. Experienced in requirements gathering, stakeholder management, and Agile methodology. Created business requirement documents and conducted gap analysis. Proficient in JIRA, Confluence, and process mapping.",
            "Data-driven business analyst specializing in product analytics and strategic planning. Expert in SQL, Power BI, and statistical analysis. Experience with A/B testing, user research, and market analysis. Strong communication and presentation skills.",
        ],
        "HR": [
            "HR professional with experience in talent acquisition, employee engagement, and performance management. Strong communication and interpersonal skills. Proficient in HRIS systems, Excel, and data analysis. Experience with labor laws, benefits administration, and organizational development.",
        ],
        "Advocate": [
            "Corporate lawyer with expertise in contract law, intellectual property, and compliance. Strong research with excellent communication and negotiation skills. Experience in litigation, legal drafting, and regulatory affairs. Proficient in legal research databases and document management systems.",
        ],
        "Arts": [
            "Creative designer with expertise in Adobe Photoshop, Illustrator, and InDesign. Strong portfolio in branding, typography, and visual communication. Experience with UI/UX design using Figma. Skilled in motion graphics, video editing, and digital illustration.",
        ],
        "Mechanical Engineer": [
            "Mechanical engineer with expertise in CAD design using SolidWorks and AutoCAD. Experience in product development, FEA analysis, and manufacturing processes. Proficient in MATLAB, C++, and project management. Strong problem-solving and analytical skills.",
        ],
        "Sales": [
            "Sales professional with proven track record in B2B enterprise sales. Expert in CRM tools like Salesforce, pipeline management, and client relationship building. Strong negotiation, presentation, and communication skills. Experience in strategic account management and business development.",
        ],
        "Health and Fitness": [
            "Health data analyst with experience in medical data analysis, clinical research, and wellness program development. Proficient in Python, SQL, and statistical analysis tools. Knowledge of health informatics, HIPAA compliance, and data visualization.",
        ],
        "Civil Engineer": [
            "Civil engineer with expertise in structural design, project management, and construction supervision. Proficient in AutoCAD, STAAD Pro, and MS Project. Experience in highway design, water resources, and environmental compliance.",
        ],
        "Electrical Engineering": [
            "Electrical engineer specializing in power systems, control engineering, and embedded systems design. Proficient in MATLAB, Simulink, and PLC programming. Experience with circuit design, PCB layout, and electrical installations.",
        ],
        "Full Stack Developer": [
            "Full stack developer with expertise in React, Node.js, and PostgreSQL. Built scalable web applications with modern JavaScript stack. Experience with Docker, AWS, and CI/CD pipelines. Strong understanding of both frontend and backend architecture.",
        ],
        "Network Security Engineer": [
            "Cybersecurity professional with expertise in penetration testing, vulnerability assessment, and incident response. Proficient in SIEM tools, firewall configuration, and network security. Certified in CISSP, CEH, and experience with OWASP, Nmap, and Wireshark.",
        ],
        "Database Administrator": [
            "Database administrator with expertise in PostgreSQL, MySQL, and Oracle. Experience in database optimization, backup and recovery, and high availability configurations. Proficient in SQL, Python scripting, and cloud database services.",
        ],
        "Automation Testing": [
            "QA automation engineer with expertise in Selenium, Cypress, and Appium. Built test automation frameworks using Java and Python. Experience with CI/CD integration, API testing, and performance testing. Proficient in BDD, JUnit, and TestNG.",
        ],
        "SAP Developer": [
            "SAP technical consultant with expertise in ABAP programming, SAP HANA, and module configuration. Experience in SAP implementation projects, data migration, and integration. Strong knowledge of business processes and ERP workflows.",
        ],
        "Hadoop Developer": [
            "Big data engineer with expertise in Hadoop ecosystem including Spark, Hive, and Kafka. Built data pipelines processing petabytes of data. Proficient in Python, Scala, and SQL. Experience with Snowflake, Databricks, and cloud data platforms.",
        ],
        "ETL Developer": [
            "ETL developer with expertise in building data integration pipelines using Informatica and Apache Spark. Experience with data warehousing, data modeling, and cloud migration. Proficient in SQL, Python, and Airflow for workflow orchestration.",
        ],
        "DotNet Developer": [
            ".NET developer with expertise in C#, ASP.NET Core, and Entity Framework. Built enterprise applications using microservices architecture. Experience with Azure, SQL Server, and Docker. Proficient in REST API development and agile methodologies.",
        ],
        "Blockchain Developer": [
            "Blockchain developer with expertise in Solidity, Ethereum, and Web3 technologies. Built decentralized applications and smart contracts. Experience with DeFi protocols, NFT platforms, and cryptographic systems. Proficient in JavaScript, Python, and Node.js.",
        ],
        "Testing": [
            "Software testing professional with expertise in manual and automated testing. Experience in test planning, test case design, and defect management. Proficient in Selenium, JIRA, and SQL. Knowledge of Agile testing practices and CI/CD integration.",
        ],
        "Operations Manager": [
            "Operations manager with expertise in process optimization, team leadership, and strategic planning. Experience with Lean Six Sigma, ERP systems, and project management. Strong analytical skills with proficiency in Excel, SQL, and business intelligence tools.",
        ],
    }

    resumes = []
    for category, templates in resume_templates.items():
        for template in templates:
            # Create multiple variations
            for i in range(random.randint(25, 45)):
                # Add some random variation to each resume
                extra_skills = random.choice([
                    " Additional experience with cloud computing and DevOps practices.",
                    " Strong background in agile methodology and team collaboration.",
                    " Passionate about continuous learning and professional development.",
                    " Experience with version control (Git), code reviews, and documentation.",
                    " Excellent problem-solving abilities and attention to detail.",
                    " Background in research methodology and academic publishing.",
                    " Experience working in cross-functional distributed teams.",
                    ""
                ])
                resumes.append({
                    "Category": category,
                    "Resume": template + extra_skills,
                })

    df = pd.DataFrame(resumes)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(RESUMES_CSV, index=False, encoding='utf-8')
    print(f"✅ Generated {len(df)} resumes across {df['Category'].nunique()} categories")
    print(f"   Category distribution:")
    for cat, count in df['Category'].value_counts().head(10).items():
        print(f"     {cat}: {count}")
    print(f"   Saved to: {RESUMES_CSV}")
    return df


def main():
    """Run complete data preparation pipeline."""
    print("=" * 60)
    print("  AI Resume Analyzer — Data Preparation Pipeline")
    print("=" * 60)
    print()

    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "sample_resumes"), exist_ok=True)

    # Generate datasets
    print("📊 Step 1: Generating resume dataset...")
    resumes_df = generate_resume_dataset()
    print()

    print("📋 Step 2: Generating job descriptions...")
    jobs_df = generate_job_descriptions()
    print()

    # Verify skills database
    print("🔧 Step 3: Verifying skills database...")
    if os.path.exists(SKILLS_DB_JSON):
        with open(SKILLS_DB_JSON, 'r', encoding='utf-8') as f:
            skills_db = json.load(f)
        total_skills = sum(len(skills) for skills in skills_db["categories"].values())
        print(f"   ✅ Skills database: {total_skills} skills across {len(skills_db['categories'])} categories")
    else:
        print("   ⚠️ Skills database not found!")

    print()
    print("=" * 60)
    print("  ✅ Data preparation complete!")
    print(f"  📁 Resumes: {len(resumes_df)} entries")
    print(f"  📁 Job Descriptions: {len(jobs_df)} entries")
    print("=" * 60)


if __name__ == "__main__":
    main()
