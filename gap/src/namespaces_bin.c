
#include        "memmgr.h"              /* Bag, NewBag, T_STRING, .. */
#include        "eval.h"                /* evaluator main dispatcher       */

#include		"idents.h"

#include        "tables.h"
#include        "plist.h"
#include        "integer.h"             



Bag HdPrivatePackagesStack;
UInt privatePackageTop;
UInt PRIVATE_PACKAGES_LEN = 16;


Obj PushPrivatePackage ( Obj hdNS ) {
    assert(hdNS != 0);
    if ( HdPrivatePackagesStack == 0 ) {
		// Put the HdPrivatePackagesStackin the global list so that garbage
		// collection will know it's an active bag
		InitGlobalBag(&HdPrivatePackagesStack, "HdPrivatePackagesStack");
        HdPrivatePackagesStack = NewBag(T_LIST, SIZE_PLEN_PLIST(PRIVATE_PACKAGES_LEN));
        SET_LEN_PLIST(HdPrivatePackagesStack, 1);
    }
    privatePackageTop++;
    AssPlist(HdPrivatePackagesStack, privatePackageTop, hdNS);
    return hdNS;
 }

Obj PopPrivatePackage (void) {
    if( privatePackageTop > 0 ) {
        Obj hdPopped = ELM_PLIST(HdPrivatePackagesStack, privatePackageTop);
        SET_ELM_PLIST(HdPrivatePackagesStack, privatePackageTop, 0);
		privatePackageTop--;			// don't do this inside macro call
        SET_LEN_PLIST(HdPrivatePackagesStack, privatePackageTop);
        return hdPopped;
    }
    else return Error("No package on top of stack", 0, 0);
}

Bag	FindInPrivatePackages(char * name, int mode_rd) {
    Bag        hdIdent, hdNS;
    UInt       i,  k;

	hdIdent = 0;
	if(privatePackageTop) {
		if ( mode_rd ) { 
			for ( i = privatePackageTop; i > 0; --i ) {
	    		hdNS = PTR_BAG(HdPrivatePackagesStack)[i];
				k = TableLookup(hdNS, name, OFS_IDENT);
				hdIdent = PTR_BAG(hdNS)[k];
				if (hdIdent != 0) break;
			}
		} else {
   			hdNS = PTR_BAG(HdPrivatePackagesStack)[privatePackageTop];
			k = TableLookup(hdNS, name, OFS_IDENT);
			hdIdent = PTR_BAG(hdNS)[k];
			if ( hdIdent == 0 ) hdIdent = TableAddIdent(hdNS, k, name);
		}
	}
	return hdIdent;
}

