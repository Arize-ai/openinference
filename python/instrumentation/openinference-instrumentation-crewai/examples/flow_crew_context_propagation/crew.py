"""
crew.py — mirrors production crew.py

Changes from production:
  - config.yaml loading removed; OPENAI_API_KEY read from env
  - agents_config / tasks_config YAML inlined as strings (same content)
"""

import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool
from pydantic import BaseModel, Field

llm = LLM(
    model="gpt-3.5-turbo",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
)


class Report(BaseModel):
    subject: str
    content: str
    customer_role: str
    company: str = Field(
        ...,
        description="The cleaned company name without corporate suffixes like 'Inc.', 'LLC', or 'Corp.'",
    )


@CrewBase
class CompanyResearchCrew:
    """CompanyResearchCrew is responsible for researching Lead companies and generating reports."""

    @agent
    def researcher(self) -> Agent:
        return Agent(
            role="Company Research Agent",
            goal="Gather comprehensive and accurate information about company {company}",
            backstory=(
                "You are a seasoned strategist with a unique blend of business acumen and "
                "product knowledge. Your expertise lies in deeply understanding {company}'s "
                "operations, challenges, and goals, then translating those insights into "
                "compelling narratives about how Docusign products can provide tangible value. "
                "You don't just gather facts; you synthesize them to create a strategic "
                "perspective that informs our marketing outreach. "
                "You have access to various data points {company}, website {email_domain}, "
                "industry {industry}, company's employee size {company_size}, its country "
                "{country}, annual revenue {annual_revenue} and use those to craft a "
                "comprehensive company profile."
            ),
            llm=llm,
            verbose=True,
            max_execution_time=180,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            role="Content Personalizer",
            goal=(
                "Craft engaging, customized marketing content using researcher insights about "
                "{company} and {industry}, highlighting relevant pain points and aligning them "
                "with key DocuSign solutions to drive clear, compelling impact."
            ),
            backstory=(
                "With a strong background in content personalization and industry analysis, "
                "you excel at crafting messages that resonate with each company's unique "
                "interests and needs, particularly those of the {job_level}. You leverage "
                "research data and industry insights to create compelling narratives that "
                "demonstrate how docusign solutions address specific challenges and drive value."
            ),
            llm=llm,
            verbose=True,
            max_execution_time=180,
        )

    @task
    def company_research_task(self) -> Task:
        return Task(
            description=(
                "Conduct a comprehensive research about the company associated with the email "
                "domain {email_domain}, which operates in the {industry} sector. "
                "The lead's title is {title} and their job level is {job_level}, department "
                "{department}. Company's annual revenue is {annual_revenue} and they are "
                "located in {country}.\n\n"
                "Instructions for company name and research approach:\n"
                "**PERSONAL EMAIL DOMAINS TO REJECT:**\n"
                "- gmail.com, yahoomail.com, outlook.com, hotmail.com, yahoo.com, aol.com, "
                "icloud.com, protonmail.com, etc.\n\n"
                "**IF {email_domain} is a personal email domain:**\n"
                "- DO NOT use any web scraping tools or external research tools\n"
                "- DO NOT infer, guess, or hallucinate a company name from the email domain\n"
                "- Use ONLY the provided company name {company} as-is (clean it of suffixes if needed)\n"
                "- If {company} is 'Not Provided', 'Unknown', 'N/A', empty, or similar, leave "
                "the company name as empty\n"
                "- Skip all web research and proceed with basic analysis using only the provided information\n\n"
                "**IF {email_domain} is NOT a personal email domain:**\n"
                "- Use the Read website content tool to scrape "
                "https://en.wikipedia.org/wiki/Arrow_Electronics for company background.\n"
                "- If the web scraping tool errors out or doesn't return company information, "
                "retry up to 2 times maximum\n"
                "- If the website content reading did not error out, but the web site content "
                "is not good for using, do not perform any more web site content reading activity\n"
                "- If tool fails twice, abandon tool usage and proceed with available information, "
                "DO NOT use the web scraping tool again for this task\n"
                "- Continue with the research using only the provided company information\n\n"
                "Additionally, from the provided company name {company}, which may be messy or "
                "contain corporate suffixes like 'Inc.', 'LLC', or 'Corp.', you must identify "
                "and extract the clean, official company name suitable for professional communication."
            ),
            expected_output=(
                "A sophisticated response that includes research findings, audience analysis, "
                "and strategic content, tailored for the audience persona based on {title} and "
                "their job level is {job_level} associated with the company {company}.\n"
                "Instructions: Follow these steps to create a content\n"
                "STEP 1. Research and Scrape: use the scrape tool to find recent news articles, "
                "'About Us' pages, and financial reports for the company associated with "
                "{company} and {email_domain} use the website scraping tool to extract the text "
                "content from these pages to understand their current strategic goals, challenges, "
                "and objectives.\n"
                "STEP 2. Understand: Based upon the research, determine what company {company} "
                "does, what they sell and how they generate revenue.\n"
                "STEP 3. Analyze: Based upon the research and an understanding of company "
                "{company}'s industry, determine what agreement types are important to company "
                "{company}.\n"
                "STEP 4. In your final output, ensure the 'company' field in the pydantic model "
                "is assigned the cleaned and professionally addressable company name."
            ),
            agent=self.researcher(),
            tools=[ScrapeWebsiteTool(max_usage_count=5)],
            output_pydantic=Report,
        )

    @task
    def product_understanding_task(self) -> Task:
        return Task(
            description=(
                "Familiarize yourself with the full range of DocuSign products and services, "
                "including their features, benefits, and use cases. "
                "Understand how these products can be applied to various industries and business "
                "challenges. Use the Read website content tool to scrape "
                "https://en.wikipedia.org/wiki/DocuSign to explore the product offerings in detail."
            ),
            expected_output=(
                "A comprehensive summary that includes:\n"
                "- A list of DocuSign products and services, along with their key features and benefits.\n"
                "- An explanation of how each product can address common business challenges across "
                "different industries.\n"
                "- Specific examples of use cases where DocuSign products have been successfully implemented.\n"
                "- Insights into how these products can be tailored to meet the needs of companies "
                "in the {industry} sector."
            ),
            agent=self.reporting_analyst(),
            tools=[ScrapeWebsiteTool(max_usage_count=5)],
            name="Product Understanding Task",
        )

    @task
    def content_personalizer_task(self) -> Task:
        """Creates the content personalizer task."""
        return Task(
            description=(
                "Review the research findings and expand each topic into a full section for a report. "
                "Provide a strategic perspective on how docusign products {docusign_products} and "
                "docusign industry solutions {docusign_solutions} can add value to the company's "
                "{company} operations. Tailor the content to the company {company}'s background, "
                "industry, and trends.\n\n"
                "Based on the fact that the company {company} operates in the {industry} industry, "
                "identify and describe 2-3 key ways Docusign solutions can help them solve common "
                "challenges specific to the {industry} sector. Frame these as benefits relevant to "
                "{company}'s potential operational goals.\n\n"
                "** Tool Instructions **\n"
                "- If the web scraping tool errors out or doesn't return company information, "
                "retry up to 2 times maximum\n"
                "- If tool fails twice, abandon tool usage and proceed with available information, "
                "DO NOT use them again."
            ),
            expected_output=(
                "A detailed, personalized report that includes:\n"
                "- Extract key themes or focus areas that resonate most with the audience persona "
                "based on {title} and {job_level}.\n"
                "- Identify the specific pain points this company/industry is experiencing and how "
                "docusign products {docusign_products} could address those needs.\n"
                "- An analysis of relevant industry solutions and trends, highlighting how DocuSign "
                "aligns with the {company}'s needs.\n"
                "- A clear value proposition showing how DocuSign's offerings can solve the "
                "company's problems.\n"
                "customer role: Identify a customer role of the lead based on job level "
                "{job_level} and title {title}, department {department}"
            ),
            agent=self.reporting_analyst(),
            tools=[ScrapeWebsiteTool(max_usage_count=5)],
            output_pydantic=Report,
            name="Content Personalizer Task",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DocusignOutreachAi crew."""
        return Crew(
            name=self.__class__.__name__,
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            full_output=True,
        )
