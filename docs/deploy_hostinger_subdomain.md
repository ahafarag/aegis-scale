# Publishing Live Demo on `aegis.zenithcodigo.com` (AWS Serverless + Hostinger DNS)

This guide deploys the **AegisScale** FastAPI + HiGHS MILP simulator onto **AWS Lambda Container Image + API Gateway HTTP API** (`AegisLiveDemoStack`), which **scales to \$0.00 when idle**, enforces hard wallet-protection limits (`ReservedConcurrentExecutions = 10` and `15 req/s` API Gateway throttling), and binds to **`https://aegis.zenithcodigo.com`** via **Hostinger hPanel DNS**.

---

## Prerequisites
1. Docker Desktop running locally (used by AWS CDK to build `Dockerfile.lambda`).
2. AWS CLI signed in (`aws login` or `aws configure`) in your preferred European region (e.g., `eu-west-1` Dublin or `eu-south-2` Spain).
3. Node.js installed (`npm install -g aws-cdk`).

---

## Step 1: Request Free TLS Certificate in AWS ACM (`eu-west-1`)

Run this single AWS CLI command to request a free public TLS certificate for `aegis.zenithcodigo.com`:

```powershell
aws acm request-certificate `
  --domain-name aegis.zenithcodigo.com `
  --validation-method DNS `
  --region eu-west-1
```

Copy the returned `CertificateArn` (`arn:aws:acm:eu-west-1:YOUR_ACCOUNT:certificate/...`), then inspect the DNS validation record:

```powershell
aws acm describe-certificate `
  --certificate-arn YOUR_CERT_ARN `
  --region eu-west-1 `
  --query "Certificate.DomainValidationOptions[0].ResourceRecord"
```

It will output a `Name` (e.g., `_a1b2c3d4.aegis.zenithcodigo.com.`) and a `Value` (e.g., `_x9y8z7.acm-validations.aws.`).

---

## Step 2: Add the Validation CNAME in Hostinger hPanel

1. Log in to **Hostinger hPanel** -> **Domains** -> **`zenithcodigo.com`** -> **DNS / Nameservers**.
2. Add a new **CNAME** record:
   * **Type:** `CNAME`
   * **Name (Host):** `_a1b2c3d4.aegis`  
     *(Important Hostinger Rule: Strip `.zenithcodigo.com.` from the end of `Name`! Hostinger automatically appends `.zenithcodigo.com`. Paste only the prefix ending in `.aegis`.)*
   * **Points to (Target):** `_x9y8z7.acm-validations.aws` *(strip trailing dot if Hostinger prompts)*
   * **TTL:** `300` (or default `14400`)
3. Wait 2–5 minutes until certificate status turns `ISSUED`:
   ```powershell
   aws acm wait certificate-validated --certificate-arn YOUR_CERT_ARN --region eu-west-1
   ```

---

## Step 3: Deploy `AegisLiveDemoStack` via AWS CDK

From `d:\aegis-scale`:

```powershell
$env:CDK_DEFAULT_REGION = "eu-west-1"
npx cdk bootstrap --config infra/cdk.json
npx cdk deploy AegisLiveDemoStack `
  --config infra/cdk.json `
  -c domain_name=aegis.zenithcodigo.com `
  -c cert_arn=YOUR_CERT_ARN
```

At the end of deployment, CDK prints two outputs:
* `AegisLiveDemoStack.DefaultHttpApiUrl = https://xxxxxxxx.execute-api.eu-west-1.amazonaws.com`
* `AegisLiveDemoStack.RegistrarCnameTarget = d-yyyyyyyyyy.execute-api.eu-west-1.amazonaws.com`

---

## Step 4: Point `aegis.zenithcodigo.com` in Hostinger DNS

In **Hostinger hPanel -> Domains -> `zenithcodigo.com` -> DNS / Nameservers**, add the subdomain CNAME:
* **Type:** `CNAME`
* **Name (Host):** `aegis`
* **Points to (Target):** `d-yyyyyyyyyy.execute-api.eu-west-1.amazonaws.com` *(exact value of `RegistrarCnameTarget`)*
* **TTL:** `300`

Within 1–3 minutes, **`https://aegis.zenithcodigo.com`** will serve the live **AegisScale Executive FinOps & EKS Karpenter Control Plane** over HTTPS!
