import { NextResponse } from "next/server"
import { auth0, isAuth0ClientReady } from "./lib/auth0"

export async function middleware(request) {
	// Redirect the root path to the chat page
	if (request.nextUrl.pathname === "/") {
		const { origin } = new URL(request.url)
		return NextResponse.redirect(`${origin}/chat`)
	}

	if (process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost") {
		// In self-host mode, authentication is handled by a static token,
		// so we don't need Auth0's session middleware.
		return NextResponse.next()
	}

	if (!isAuth0ClientReady) {
		return new NextResponse(
			"Auth0 is not configured: set AUTH0_DOMAIN (tenant host, e.g. dev-xxx.us.auth0.com) or AUTH0_ISSUER_BASE_URL in .env.local, plus AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_SECRET. Or set NEXT_PUBLIC_ENVIRONMENT=selfhost.",
			{ status: 503, headers: { "content-type": "text/plain; charset=utf-8" } }
		)
	}

	const authRes = await auth0.middleware(request)

	// authentication routes — let the middleware handle it
	if (request.nextUrl.pathname?.startsWith("/auth")) {
		return authRes
	}

	const { origin } = new URL(request.url)
	const session = await auth0.getSession(request)

	// user does not have a session — redirect to login
	if (!session) {
		return NextResponse.redirect(`${origin}/auth/login`)
	}

	return authRes
}

export const config = {
	matcher: [
		/*
		 * Match all request paths except for the ones starting with:
		 * - _next/static (static files)
		 * - _next/image (image optimization files)
		 * - favicon.ico, sitemap.xml, robots.txt (metadata files)
		 * - api (API routes)
		 * - PWA files (manifest, icons, service worker, workbox)
		 * - .png and .svg files (static images)
		 */
		"/((?!_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt|api|manifest.json|manifest.webmanifest|sw.js|workbox-.*\\.js$|.*\\.png$|.*\\.svg$).*)"
	]
}
