import { clerkMiddleware } from '@clerk/nextjs/server';

// Use the basic clerk middleware
export default clerkMiddleware();

// Only apply middleware to /chat and /dashboard routes
export const config = {
  matcher: ['/chat(.*)', '/dashboard(.*)'],
};
