/**
 * Minimal ambient type declarations for commander
 * Only the subset used by this plugin's CLI registration.
 */
declare module "commander" {
  export class Command {
    command(nameAndArgs: string, opts?: Record<string, unknown>): Command;
    description(str: string): Command;
    option(flags: string, description?: string, defaultValue?: string): Command;
    requiredOption(flags: string, description?: string, defaultValue?: string): Command;
    argument(name: string, description?: string): Command;
    action(fn: (...args: any[]) => void | Promise<void>): Command;
  }
}
