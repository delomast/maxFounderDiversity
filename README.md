    
## Self-supervised Learning of Optimum Contributions: An Application in Maximizing Genetic Variation
![example workflow](https://github.com/delomast/maxFounderDiversity/actions/workflows/ci.yml/badge.svg?event=push)

Algorithm to choose which populations to sample broodstock from and in what proportions to maximize genetic diversity (expected heterozygosity) in the population of offspring given a set of populations that could be sampled from (with known allele frequencies at a common set of loci).

Developed with the application of creating base populations for aquaculture breeding programs in mind.

<details>
<summary>Motivation</summary>
Diversity of traits in living organisms is controlled by inherited genes. 
Therefore, the success of selective breeding tasks using genetic data predominant in the agricultural sciences is highly correlated with the degree of genetic variants present in the founding populations used for that breeding program. Today, genetic data can be digitally synthesized broadening the genetic variation range that can be obtained for founding a breeding program. A large number of populations, say $n \ge 50$ can now be surveyed as possible candidates that could be in the founder set.


Given a number of populations, $n$, we typically want to select $k\le n$ founding populations for a breeding program in a way that will maximize the genetic variation (or minimize the co-ancestry) of their offspring. For each $1 \le i\le n$ population, available information is a genomic dataset of allele frequencies for $l$ loci. 

</details>

<details>
  <summary>Problem Statement</summary>

  It is usually assumed that all available $n$ populations can be combined and sampled for use in the breeding program, that is, we choose $k=n$ populations. This plan was a sensible about two decades ago when genotyping was expensive. In contrast, in recent times, large-scale genotyping data is cheaper to obtain.  However, choosing broodstock from all of the populations is likely redundant (diminishing returns).

  For optimum cost-effective planning, we would like to evaluate each possible $k$ founding set, where $1\le k\le n$, and pick a $k$ combination at which a further increase in $k$, starts to add little to the average genetic diversity in the group. For example, given a dataset of $n=20$ populations, we may find that choosing between $k=5$ to $k=8$ populations is sufficient to create a successful breeding program.

</details>

<details>
  <summary>Objective</summary>
  Here we present a self-supervised learning algorithm for efficiently solving large-scale problems of this nature. 

  
  Our tool assists with making the decision of which $k$ combination of the $n$ populations to choose and the relative proportion (or number) of broodstock from each? 
  
  
  Given known allele frequencies for $l$ loci in $n$ available populations. The goal of our learning algorithm is to both select a subset $k \le n$ populations and determine the number of individuals to select from each of $k$ populations in a way that maximizes the genetic variation of the given group, with the least diminishing return.
</details>



#### Installing
Clone this repo. From the root path of this repo on your local machine:

Install a python virtual environment (See: <a>https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/ </a>).

Install required python packages with the requirements.txt file `pip install -r requirements.txt`

#### Command Line Interface CLI
Optional. To test the tool. Run: `python ssltest.py` 

#### Web Frontend
In your terminal, run: 
`
flask --app sslview run --debug --host=0.0.0.0  
`
to access the tool in form of a user-friendly web application.
<details>
  <summary> Quick Start (Web Frontend) </summary>
  <div>
      To start learning. Choose a configuration. Upload your genetic dataset of $n$ populations with allele frequencies. Header of dataset should be of the common form below: <br><br>
    <table>
      <thead>
        <tr>
          <th>CHROM</th>
          <th>POS</th>
          <th>N_ALLELES</th>
          <th>N_CHR</th>
          <th>{ALLELE:FREQ}</th>
        </tr>
      </thead>
    </table>
    where <strong>CHROM</strong> is a chromosome name, <strong>POS</strong> is a position (loci) in that chromosome, <strong>N_ALLELES</strong> is the number of alleles, <strong>N_CHR</strong> is related to the sample size that was used to calculate the allele frequencies, <strong>{ALLELE:FREQ}</strong> is the dictionary of alleles and their frequencies. 
    
Each line of the $n$ input files should have the same chromosome name and position for all populations. We adopt this particular format of input file, since it can be easily generated from common genotype file formats with existing, widely used software.
  
</div>
</details>











