# Make the app auto-update (free, hands-off)

This puts the app on **GitHub Pages** and rebuilds it **twice a day automatically**
with the latest match results + xG — even when your computer is off. No coding.

## One-time setup (~10 minutes)

**1. Create a free GitHub account** — https://github.com/signup

**2. Create a repository**
- Top-right **“+” → New repository**
- Repository name: `worldcup` (or anything)
- Set it to **Public**
- Click **Create repository**

**3. Upload the project files**
- On the new repo page, click the **“uploading an existing file”** link.
- Open your folder `…\Desktop\Claude Output\World Cup Model\` and drag in these 4 files:
  `build_context.py`, `build_ratings.py`, `groups.py`, `export_web.py`
- Also drag in the whole **`web`** folder.
- Click **Commit changes**.

**4. Add the automation file** (the part that rebuilds daily)
- Click **Add file → Create new file**.
- In the filename box type exactly:  `.github/workflows/update.yml`
- Paste the entire contents of the `update.yml` file (it’s in your folder under
  `.github\workflows\update.yml`).
- Click **Commit changes**.

**5. Turn on Pages**
- Click **Settings** (top) → **Pages** (left menu).
- Under **Build and deployment → Source**, choose **GitHub Actions**.

**6. Done.** Open the **Actions** tab — you’ll see “Update predictions” running.
When it finishes (green check), your live, auto-updating site is at:
```
https://YOUR-USERNAME.github.io/worldcup/
```
Share that link with friends. On a phone: browser menu → **Add to Home Screen**.

## After setup
- It rebuilds automatically at **8am and 8pm UTC every day** from the latest data.
- Force an update now: **Actions → Update predictions → Run workflow**.
- You never have to touch it again — the link always shows current numbers.

> Note: this GitHub Pages link replaces the manual Netlify one as your shareable,
> always-current address. (You can delete the Netlify site if you made one.)
